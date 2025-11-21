# streamlit_emg_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import io
import zipfile
from datetime import datetime

st.set_page_config(page_title="EMG Review Dashboard", layout="wide", initial_sidebar_state="expanded")

# -----------------------
# Helper utilities
# -----------------------
TIME_CANDIDATES = ['time', 't', 'seconds', 'sec', 'timestamp', 'time (s)', 'time_s']
EMG_RMS_CANDIDATES = ['emgrms', 'rms', 'emg_rms', 'channel_1_rms', 'rms_value', 'rms(mv)', 'rms_mv']
EMG_RAW_CANDIDATES = ['emg_raw', 'emgraw', 'emg', 'ch1', 'channel1', 'channel_1', 'raw', 'raw_emg', 'channel_1_mv']
BASELINE_SHEETS = ['baseline', 'base', 'rest']

def lower_list(lst):
    return ["" if x is None else str(x).lower() for x in lst]

def find_first_match(names_lower, candidates):
    for cand in candidates:
        for idx, name in enumerate(names_lower):
            if cand in name:
                return idx
    return None

def safe_read_excel(file_stream):
    try:
        xls = pd.ExcelFile(file_stream)
        return xls
    except Exception as e:
        raise RuntimeError(f"Unable to read Excel file: {e}")

def ensure_numeric(arr):
    a = np.array(arr)
    if a.dtype.kind in 'OSU':  # object/string
        a = pd.to_numeric(a, errors='coerce').astype(float)
    else:
        a = a.astype(float)
    return a

def movmean(x, win):
    if win <= 1:
        return x
    w = np.ones(win) / win
    out = np.convolve(x, w, mode='same')
    return out

def compute_envelope_from_raw(raw, fs, bp_low, bp_high, bp_order, env_win_samples):
    raw = np.asarray(raw, dtype=float)
    raw = raw - np.nanmean(raw)
    nyq = fs / 2.0
    low = bp_low / nyq
    high = bp_high / nyq
    if low <= 0: low = 1e-6
    if high >= 1: high = 0.999999
    if low >= high:
        spiky = raw.copy()
    else:
        try:
            b, a = signal.butter(bp_order, [low, high], btype='band')
            spiky = signal.filtfilt(b, a, raw)
        except Exception:
            b, a = signal.butter(bp_order, [low, high], btype='band')
            spiky = signal.lfilter(b, a, raw)
    squared = np.abs(spiky) ** 2
    env = np.sqrt(movmean(squared, env_win_samples))
    return spiky, env

def synthesize_spiky_from_rms(rms, fs, bp_low, bp_high, bp_order, env_win_samples, random_seed=None):
    N = len(rms)
    rng = np.random.default_rng(random_seed)
    noise = rng.standard_normal(N)
    nyq = fs / 2.0
    low = max(bp_low / nyq, 1e-6)
    high = min(bp_high / nyq, 0.999999)
    try:
        b, a = signal.butter(bp_order, [low, high], btype='band')
        w = signal.filtfilt(b, a, noise)
    except Exception:
        b, a = signal.butter(bp_order, [low, high], btype='band')
        w = signal.lfilter(b, a, noise)
    mov = np.sqrt(movmean(w**2, env_win_samples))
    mov[mov == 0] = np.finfo(float).eps
    w_scaled = w * (rms / mov)
    env = rms
    return w_scaled, env

# -----------------------
# UI - Sidebar controls
# -----------------------
st.sidebar.title("EMG Review Dashboard")
st.sidebar.write("Upload an Excel file with sheet(s) containing EMG data (time, raw, or RMS).")

uploaded = st.sidebar.file_uploader("Excel (.xlsx/.xls)", type=["xlsx", "xls"])
st.sidebar.markdown("---")

st.sidebar.markdown("### Processing options")
sampling_rate = st.sidebar.number_input("Sampling Rate (Hz)", value=1000, min_value=1, step=1)
rms_window_ms = st.sidebar.slider("Envelope/RMS window (ms)", min_value=10, max_value=500, value=50)
env_win_samples = max(1, int((rms_window_ms / 1000.0) * sampling_rate))
st.sidebar.write(f"Envelope window ≈ {env_win_samples} samples")

st.sidebar.markdown("**Bandpass filter (for spiky)**")
apply_bp = st.sidebar.checkbox("Apply bandpass", value=True)
col1, col2 = st.sidebar.columns(2)
with col1:
    bp_low = st.number_input("f_low (Hz)", value=20.0, min_value=0.1, format="%.1f")
with col2:
    bp_high = st.number_input("f_high (Hz)", value=450.0, min_value=bp_low + 1.0, format="%.1f")
bp_order = st.sidebar.selectbox("Filter order (Butterworth)", options=[2, 4, 6], index=1)

st.sidebar.markdown("---")
process_mode = st.sidebar.radio("Process", options=["Selected sheet only", "Process ALL sheets"], index=0)
st.sidebar.markdown("---")
run_button = st.sidebar.button("Process & Plot")

# -----------------------
# Main panel - header
# -----------------------
st.title("EMG Review — Spiky + Envelope Dashboard")
st.write("Visualize EMG 'spiky' traces and RMS envelopes. Can synthesize spiky from RMS-only sheets, and export results.")

if uploaded is None:
    st.info("Upload an Excel file using the sidebar to begin.")
    st.stop()

# -----------------------
# Read Excel and sheet selection
# -----------------------
try:
    xls = safe_read_excel(uploaded)
    sheet_names = xls.sheet_names
except Exception as e:
    st.error(str(e))
    st.stop()

st.sidebar.write(f"Loaded file with {len(sheet_names)} sheet(s).")
sheet = st.selectbox("Select a sheet to preview", options=sheet_names)

# preview the chosen sheet
try:
    df_preview = pd.read_excel(xls, sheet_name=sheet)
except Exception as e:
    st.error(f"Unable to read sheet {sheet}: {e}")
    st.stop()

st.write("### Preview (first rows)")
st.dataframe(df_preview.head())

# Column detection / overrides
st.write("### Column detection (automatic — override if needed)")
cols = df_preview.columns.tolist()
cols_lower = lower_list(cols)

time_guess = find_first_match(cols_lower, TIME_CANDIDATES)
rms_guess = find_first_match(cols_lower, EMG_RMS_CANDIDATES)
raw_guess = find_first_match(cols_lower, EMG_RAW_CANDIDATES)

# show options with None first
col_time = st.selectbox("Time column (detected)", options=[None] + cols, index=0 if time_guess is None else (cols.index(cols[time_guess]) + 1))
col_rms = st.selectbox("EMG RMS column (detected)", options=[None] + cols, index=0 if rms_guess is None else (cols.index(cols[rms_guess]) + 1))
col_raw = st.selectbox("EMG Raw column (detected)", options=[None] + cols, index=0 if raw_guess is None else (cols.index(cols[raw_guess]) + 1))

# -----------------------
# Processing engine
# -----------------------
def process_sheet_table(table: pd.DataFrame, time_col=None, raw_col=None, rms_col=None,
                        fs=1000, bp_low_hz=20.0, bp_high_hz=450.0, bp_order=4, env_win_samples=50):
    T = table.copy()
    names = T.columns.tolist()
    names_lower = lower_list(names)

    # time column selection
    if time_col is None:
        idx = find_first_match(names_lower, TIME_CANDIDATES)
        time_col_use = names[idx] if idx is not None else names[0]
    else:
        time_col_use = time_col

    if rms_col is None:
        idx = find_first_match(names_lower, EMG_RMS_CANDIDATES)
        rms_col_use = names[idx] if idx is not None else None
    else:
        rms_col_use = rms_col

    if raw_col is None:
        idx = find_first_match(names_lower, EMG_RAW_CANDIDATES)
        raw_col_use = names[idx] if idx is not None else None
    else:
        raw_col_use = raw_col

    # Extract time
    Time = T[time_col_use] if time_col_use in T.columns else pd.Series(np.arange(len(T)))
    if np.issubdtype(Time.dtype, np.datetime64):
        Time = (Time - Time.iloc[0]).dt.total_seconds()
    Time = ensure_numeric(Time)
    if np.all(np.isnan(Time)):
        Time = np.arange(len(T)) / float(fs)

    # Extract RMS and raw if present
    EMGrms = None
    if rms_col_use and rms_col_use in T.columns:
        EMGrms = ensure_numeric(T[rms_col_use].values)
    EMGraw = None
    if raw_col_use and raw_col_use in T.columns:
        EMGraw = ensure_numeric(T[raw_col_use].values)

    # Remove rows with NaNs in essential columns
    mask = ~np.isnan(Time)
    if EMGrms is not None:
        mask = mask & ~np.isnan(EMGrms)
    if EMGraw is not None:
        mask = mask & ~np.isnan(EMGraw)

    Time = Time[mask]
    if EMGrms is not None:
        EMGrms = EMGrms[mask]
    if EMGraw is not None:
        EMGraw = EMGraw[mask]

    # Estimate sampling frequency from time vector
    if len(Time) >= 3:
        dt = np.median(np.diff(Time))
        fs_est = 1.0 / dt if dt > 0 else fs
    else:
        fs_est = fs

    # adjust bandpass if fs small
    local_bp_low = float(bp_low_hz)
    local_bp_high = float(bp_high_hz)
    if fs_est <= 2 * local_bp_high:
        local_bp_high = max((fs_est / 2.0) - 1.0, local_bp_low + 1.0)

    # Build spiky and env
    if EMGraw is not None and len(EMGraw) >= 3:
        spiky, env = compute_envelope_from_raw(EMGraw, fs_est, local_bp_low, local_bp_high, bp_order, env_win_samples)
    elif EMGrms is not None:
        spiky, env = synthesize_spiky_from_rms(EMGrms, fs_est, local_bp_low, local_bp_high, bp_order, env_win_samples, random_seed=42)
    else:
        raise ValueError("Sheet contains neither raw EMG nor RMS columns we can detect.")

    return {
        'time': np.array(Time, dtype=float),
        'spiky': np.array(spiky, dtype=float),
        'env': np.array(env, dtype=float)
    }

# -----------------------
# Plotting utility
# -----------------------
def plot_polished_tiles(data_list, title_prefix="EMG"):
    n = len(data_list)
    if n == 0:
        st.warning("No data to plot.")
        return

    global_max = 0.0
    for d in data_list:
        global_max = max(global_max, np.nanmax(np.abs(np.concatenate([d['spiky'], d['env']] ))))

    if global_max <= 0:
        global_max = 1.0
    Ymax = np.ceil(global_max / 0.5) * 0.5
    YL = (-Ymax, Ymax)
    y_ticks = np.arange(YL[0], YL[1] + 1e-9, 0.5)

    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), squeeze=False)
    axes = axes.flatten()

    for ax in axes[n:]:
        ax.axis('off')

    for i, d in enumerate(data_list):
        ax = axes[i]
        Time = d['time']
        spiky = d['spiky']
        env = d['env']
        sheet = d.get('sheet_name', f"Sheet {i+1}")

        sheet_lower = sheet.lower()
        if any(key in sheet_lower for key in BASELINE_SHEETS):
            xEnd = 5
            xticks = np.arange(0, xEnd + 0.1, 0.5)
        else:
            xEnd = 15
            xticks = np.arange(0, xEnd + 0.1, 1.0)

        order = np.argsort(Time)
        Time_s = Time[order]
        sp = spiky[order]
        en = env[order]

        if Time_s[-1] < xEnd:
            Time_s = np.concatenate([Time_s, [xEnd]])
            sp = np.concatenate([sp, [sp[-1]]])
            en = np.concatenate([en, [en[-1]]])

        ax.fill_between(Time_s, -en, en, color=(1.0, 0.6, 0.2), alpha=0.18, edgecolor='none')
        ax.plot(Time_s, sp, color=(0.0, 0.4470, 0.7410, 0.65), linewidth=0.6)
        ax.plot(Time_s, en, color=(0.85, 0.325, 0.098), linewidth=1.2)
        ax.plot(Time_s, -en, color=(0.85, 0.325, 0.098, 0.6), linestyle='--', linewidth=0.9)

        ax.set_title(f"{title_prefix} – {sheet}", fontweight='bold', fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.grid(True, alpha=0.35)
        ax.set_xlim(0, xEnd)
        ax.set_ylim(YL)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='both', which='major', labelsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# -----------------------
# Run processing
# -----------------------
if run_button:
    if process_mode == "Selected sheet only":
        sheets_to_process = [sheet]
    else:
        sheets_to_process = sheet_names

    st.info(f"Processing {len(sheets_to_process)} sheet(s)...")
    processed_results = []
    errors = {}

    for sh in sheets_to_process:
        try:
            df_sh = pd.read_excel(xls, sheet_name=sh)
            if sh == sheet:
                tcol = col_time
                rcol = col_rms
                rawcol = col_raw
            else:
                tcol = None; rcol = None; rawcol = None

            data = process_sheet_table(df_sh, time_col=tcol, raw_col=rawcol, rms_col=rcol,
                                       fs=sampling_rate, bp_low_hz=bp_low, bp_high_hz=bp_high,
                                       bp_order=bp_order, env_win_samples=env_win_samples)
            data['sheet_name'] = sh
            processed_results.append(data)
        except Exception as e:
            errors[sh] = str(e)

    if len(processed_results) == 0:
        st.error("No sheets were successfully processed. See errors below.")
        st.write(errors)
    else:
        st.success(f"Processed {len(processed_results)} / {len(sheets_to_process)} sheets successfully.")
        if errors:
            st.warning("Some sheets failed to process. See details below.")
            st.write(errors)

        # Plot polished tiles
        plot_polished_tiles(processed_results, title_prefix="EMG")

        # Build export: multi-sheet Excel or ZIP of CSVs
        st.write("### Export processed results")
        export_mode = st.radio("Export format", options=["Single multi-sheet Excel (.xlsx)", "ZIP of CSVs"], index=0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if export_mode.startswith("Single"):
            out = io.BytesIO()
            # Use context manager (no writer.save())
            with pd.ExcelWriter(out, engine='openpyxl') as writer:
                for d in processed_results:
                    dfout = pd.DataFrame({
                        'time_s': d['time'],
                        'spiky': d['spiky'],
                        'env': d['env']
                    })
                    sheet_safe = (d['sheet_name'][:30]) if d.get('sheet_name') else f"Sheet_{len(d)}"
                    # Excel limits sheet names to 31 chars; ensure no invalid chars
                    sheet_safe = sheet_safe.replace('/', '_').replace('\\', '_')
                    dfout.to_excel(writer, sheet_name=sheet_safe, index=False)
            out.seek(0)
            st.download_button(
                label=f"Download processed_{timestamp}.xlsx",
                data=out,
                file_name=f"processed_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            in_memory = io.BytesIO()
            with zipfile.ZipFile(in_memory, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                for d in processed_results:
                    dfout = pd.DataFrame({
                        'time_s': d['time'],
                        'spiky': d['spiky'],
                        'env': d['env']
                    })
                    csv_bytes = dfout.to_csv(index=False).encode('utf-8')
                    filename = f"{d['sheet_name'][:80]}.csv"
                    filename = filename.replace('/', '_').replace('\\', '_')
                    zf.writestr(filename, csv_bytes)
            in_memory.seek(0)
            st.download_button(
                label=f"Download processed_{timestamp}.zip",
                data=in_memory,
                file_name=f"processed_{timestamp}.zip",
                mime="application/zip"
            )

    st.balloons()
else:
    st.write("Click **Process & Plot** in the sidebar to start processing the selected sheet(s).")
