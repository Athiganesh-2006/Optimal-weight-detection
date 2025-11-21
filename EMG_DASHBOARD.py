# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import butter, sosfiltfilt, convolve
from typing import Optional, Tuple

# --- CONFIG ---
st.set_page_config(page_title="EMG Signal Analysis Dashboard", layout="wide")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
<style>
    /* Main Streamlit container background */
    .stApp {
        background-color: #f8f9fa; 
    }
    /* Title font style */
    .st-emotion-cache-1jm6hpn {
        color: #1a1a1a;
        font-weight: 700;
    }
    /* Sidebar header */
    [data-testid="stSidebar"] .st-emotion-cache-14ymf9f {
        color: #1a1a1a;
        font-size: 1.25rem;
    }
    /* Button styling */
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Helper functions (Unchanged) ----------
# (The helper functions read_workbook, to_numeric_series, moving_rms, moving_average, 
# design_filter, and apply_sos_filter remain as they are functional and correct.)

def read_workbook(file_bytes: bytes) -> dict:
    """Read uploaded excel bytes and return dict of sheet_name -> DataFrame"""
    with io.BytesIO(file_bytes) as fh:
        xls = pd.ExcelFile(fh)
        sheets = {}
        for name in xls.sheet_names:
            df = xls.parse(name, header=0)  # assume first row = header
            sheets[name] = df
    return sheets

def to_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce values to numeric, replace non-convertible with NaN"""
    return pd.to_numeric(s, errors="coerce")

def moving_rms(arr: np.ndarray, window_samples: int) -> np.ndarray:
    """Compute moving RMS. arr may contain NaN; result has NaN where insufficient data."""
    if window_samples <= 1:
        out = np.sqrt(np.nanmean(np.square(arr), axis=0))
        return np.full_like(arr, out) if np.ndim(arr) == 1 else out
    # square, replace NaN with 0 for convolution but keep count of valid elements
    sq = np.square(np.nan_to_num(arr, nan=0.0))
    kernel = np.ones(window_samples)
    sum_sq = convolve(sq, kernel, mode='same')
    # count of valid elements per window
    valid = (~np.isnan(arr)).astype(float)
    count = convolve(valid, kernel, mode='same')
    # avoid division by zero
    with np.errstate(invalid='ignore', divide='ignore'):
        rms = np.sqrt(sum_sq / count)
    rms[count == 0] = np.nan
    return rms

def moving_average(arr: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return arr.copy()
    kernel = np.ones(window_samples) / window_samples
    arr_n = np.nan_to_num(arr, nan=0.0)
    sum_vals = convolve(arr_n, kernel, mode='same')
    valid = (~np.isnan(arr)).astype(float)
    count = convolve(valid, np.ones(window_samples), mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        ma = sum_vals / count
    ma[count == 0] = np.nan
    return ma

def design_filter(filter_type: str, cutoff: Tuple[float, Optional[float]], fs: float, order=4):
    """Return second-order-sections (sos) for butterworth filter."""
    ny = 0.5 * fs
    if filter_type == 'band':
        low, high = cutoff
        if low <= 0 or high <= 0 or high <= low:
            return None
        lown = low / ny
        highn = high / ny
        sos = butter(order, [lown, highn], btype='bandpass', output='sos')
        return sos
    else:
        c = cutoff
        if (not c) or c <= 0:
            return None
        wn = c / ny
        if wn >= 1.0:
            return None
        sos = butter(order, wn, btype='low' if filter_type == 'low' else 'high', output='sos')
        return sos

def apply_sos_filter(arr: np.ndarray, sos):
    if sos is None:
        return arr.copy()
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return arr.copy()
    x = arr.copy()
    if nan_mask.any():
        # simple linear interpolation of NaNs
        idx = np.arange(x.size)
        good = ~nan_mask
        x[nan_mask] = np.interp(idx[nan_mask], idx[good], x[good])
    y = sosfiltfilt(sos, x)
    y[nan_mask] = np.nan
    return y

# ---------- UI & Processing Logic (Revised) ----------
st.title("EMG Signal Analysis Dashboard 📈")
st.markdown("---")

uploaded_file = st.file_uploader("Choose Excel file (.xlsx/.xls)", type=['xlsx','xls'])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    try:
        sheets = read_workbook(file_bytes)
    except Exception as e:
        st.error(f"❌ Failed to read workbook: {e}")
        st.stop()

    # --- Sidebar: Sheet Selection + Params ---
    st.sidebar.header("Data Selection & Parameters")
    sheet_names = list(sheets.keys())
    sel_sheet = st.sidebar.selectbox("Select Sheet", sheet_names)

    df = sheets[sel_sheet].copy()
    st.subheader(f"Data Sheet: **{sel_sheet}**")

    # X-axis setup
    cols = df.columns.tolist()
    if len(cols) < 2:
        st.warning("Sheet must have at least two columns (X + one channel).")
        st.stop()
    x_col = cols[0]
    st.sidebar.info(f"X-axis column (time/index): **{x_col}**")
    x_series = to_numeric_series(df[x_col])
    x_is_numeric = not x_series.isna().all()

    channel_choices = cols[1:]
    sel_channels = st.sidebar.multiselect("Channels to Analyze", channel_choices, 
                                          default=channel_choices[:min(3,len(channel_choices))])
    
    # Processed Data Options
    st.sidebar.subheader("Signal Processing Settings")
    sample_rate = st.sidebar.number_input("Sampling Rate ($f_s$) (Hz)", value=2000.0, min_value=1.0, format="%.1f")
    
    # RMS Window (for the envelope)
    rms_window_ms = st.sidebar.slider("RMS Window (ms)", min_value=10.0, max_value=500.0, value=50.0, step=10.0)
    
    # Smoothing Window (for the final envelope curve)
    smooth_window_samples = st.sidebar.number_input("Smoothing Window (samples)", value=5, min_value=1, step=1)

    # --- Filter Options ---
    st.sidebar.subheader("Filters (4th Order Butterworth)")
    bandpass_enabled = st.sidebar.checkbox("Use Bandpass Filter", value=True)
    
    if bandpass_enabled:
        bp_low = st.sidebar.number_input("Low Cutoff ($f_{low}$) (Hz)", value=20.0, min_value=0.0)
        bp_high = st.sidebar.number_input("High Cutoff ($f_{high}$) (Hz)", value=450.0, min_value=0.0)
        hp_cut, lp_cut = 0, 0 # Disable cascade if bandpass is on
    else:
        hp_cut = st.sidebar.number_input("Highpass Cutoff (Hz)", value=20.0, min_value=0.0)
        lp_cut = st.sidebar.number_input("Lowpass Cutoff (Hz)", value=450.0, min_value=0.0)
        bp_low, bp_high = 0, 0

    # --- Action Button ---
    if st.sidebar.button("Process & Plot EMG Data 🚀"):

        if not sel_channels:
            st.warning("Please select at least one channel to plot.")
            st.stop()
        
        # Prepare x values
        if x_is_numeric:
            x_vals = x_series.to_numpy(dtype=float)
        else:
            x_vals = np.arange(len(df))

        fig = go.Figure()
        processed_tables = {}
        
        # Compute windows
        # MATLAB uses 50ms window for envelope, which is RMS for filtered raw signal
        rms_window_samples = max(1, int(round((rms_window_ms / 1000.0) * sample_rate)))
        smooth_window_samples = max(1, int(smooth_window_samples))

        # Design filters
        sos_hp = design_filter('high', hp_cut, sample_rate, order=4) if (not bandpass_enabled and hp_cut > 0) else None
        sos_lp = design_filter('low', lp_cut, sample_rate, order=4) if (not bandpass_enabled and lp_cut > 0) else None
        sos_bp = design_filter('band', (bp_low, bp_high), sample_rate, order=4) if bandpass_enabled else None

        # --- Processing Loop ---
        for ch in sel_channels:
            series = to_numeric_series(df[ch])
            arr = series.to_numpy(dtype=float)
            
            # 1. Detrend / Center Data (Mimicking MATLAB raw - mean(raw))
            # Since EMG is typically AC coupled, we just remove the mean.
            arr_detrended = arr - np.nanmean(arr)

            # 2. Apply Bandpass or Cascade filter
            if bandpass_enabled and sos_bp is not None:
                filtered_signal = apply_sos_filter(arr_detrended, sos_bp)
            else:
                filtered_signal = arr_detrended.copy()
                if sos_hp is not None:
                    filtered_signal = apply_sos_filter(filtered_signal, sos_hp)
                if sos_lp is not None:
                    filtered_signal = apply_sos_filter(filtered_signal, sos_lp)
            
            # The MATLAB 'spiky' signal is this filtered signal
            spiky = filtered_signal
            
            # 3. Calculate RMS Envelope (The "env" in MATLAB)
            # EMG envelope is often approximated by the RMS of the filtered, rectified signal.
            # MATLAB uses sqrt(movmean(abs(spiky).^2, envWin)), which is RMS of spiky signal.
            rms_env = moving_rms(spiky, rms_window_samples)
            
            # 4. Apply Final Smoothing (The final visible line on the MATLAB plot)
            rms_smooth = moving_average(rms_env, smooth_window_samples)
            
            # --- Plotting (Mimic MATLAB's Look) ---
            
            # Add Shaded Area (The ±Envelope patch)
            # Plotly equivalent of fill(xx, yy)
            fill_color = 'rgba(255, 102, 0, 0.18)' # Orange/red transparent fill
            
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_vals, x_vals[::-1]]),
                y=np.concatenate([rms_smooth, -rms_smooth[::-1]]),
                fill='toself',
                fillcolor=fill_color,
                line=dict(width=0),
                name=f"{ch} Envelope Area",
                hoverinfo='skip',
                legendgroup=ch,
                showlegend=True
            ))
            
            # Add Spiky/Raw Filtered Signal (The 'spiky' line)
            fig.add_trace(go.Scatter(
                x=x_vals, y=spiky, 
                mode='lines', 
                name=f"{ch} (Filtered Signal)", 
                line=dict(width=0.6, color='rgba(0, 71, 120, 0.55)'), # Blue transparent
                legendgroup=ch,
                hoverinfo='x+y',
                showlegend=True 
            ))
            
            # Add Positive RMS Envelope (The solid line)
            fig.add_trace(go.Scatter(
                x=x_vals, y=rms_smooth, 
                mode='lines', 
                name=f"{ch} (RMS Envelope)", 
                line=dict(width=1.4, color='rgb(217, 83, 25)'), # Dark Orange
                legendgroup=ch,
                hoverinfo='x+y',
                showlegend=True
            ))
            
            # Add Negative RMS Envelope (The dashed line)
            fig.add_trace(go.Scatter(
                x=x_vals, y=-rms_smooth, 
                mode='lines', 
                name=f"{ch} (Negative Envelope)", 
                line=dict(width=0.8, dash='dash', color='rgba(217, 83, 25, 0.6)'),
                legendgroup=ch,
                hoverinfo='skip',
                showlegend=False # Hide from legend for cleaner look
            ))
            
            # --- Table for Export ---
            table_df = pd.DataFrame({
                x_col: x_vals, 
                f"{ch}_raw_filtered": spiky, 
                f"{ch}_rms_envelope": rms_smooth
            })
            processed_tables[ch] = table_df

        # --- Figure Layout ---
        fig.update_layout(
            height=600, 
            template="plotly_white",
            title=dict(
                text=f"**EMG Analysis**: Sheet **{sel_sheet}**", 
                font=dict(size=20, color='#1a1a1a')
            ),
            xaxis_title=x_col + (' (s)' if x_is_numeric else ''), 
            yaxis_title="Amplitude ($\mu V$ or mV)",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                bgcolor='rgba(255,255,255,0.7)', bordercolor="#cccccc", borderwidth=1
            ),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=100, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        
        # --- Download Options ---
        st.subheader("Download Processed Data")
        
        # Combine processed tables
        combined = None
        for ch, tdf in processed_tables.items():
            if combined is None:
                combined = tdf
            else:
                combined = pd.concat([combined, tdf.drop(columns=[x_col])], axis=1)

        if combined is not None:
            col_csv, col_png = st.columns(2)
            
            # CSV Download
            csv_bytes = combined.to_csv(index=False).encode('utf-8')
            col_csv.download_button("💾 Download Processed CSV", data=csv_bytes, file_name=f"{sel_sheet}_processed.csv", mime="text/csv")
            
            # PNG Download
            try:
                # Add a high-resolution export for a professional PNG
                png_bytes = fig.to_image(format="png", width=1600, height=900, scale=2) 
                col_png.download_button("🖼️ Download Chart PNG", data=png_bytes, file_name=f"{sel_sheet}_chart.png", mime="image/png")
            except Exception:
                 col_png.info("PNG chart export requires the `kaleido` package (`pip install kaleido`).")

        # --- Summary of Analysis ---
        st.subheader("Analysis Parameters Summary")
        
        rms_window_str = f"{rms_window_ms} ms (≈ {rms_window_samples} samples)"
        smooth_window_str = f"{smooth_window_samples} samples"
        
        if bandpass_enabled:
            filter_str = f"Bandpass: {bp_low} Hz – {bp_high} Hz (4th Order Butterworth)"
        else:
            filter_str = "Filter Cascade: "
            filter_str += f"Highpass: {hp_cut} Hz. " if hp_cut > 0 else ""
            filter_str += f"Lowpass: {lp_cut} Hz." if lp_cut > 0 else ""
            if not hp_cut and not lp_cut:
                 filter_str = "No digital filtering applied."

        st.table(pd.DataFrame({
            'Parameter': ['Sampling Rate', 'RMS Window', 'Smoothing Window', 'Digital Filter'],
            'Value': [f"{sample_rate} Hz", rms_window_str, smooth_window_str, filter_str]
        }))

else:
    st.info("⬆️ Upload an Excel file to begin analyzing your EMG data.")
    st.markdown("""
    **Workflow Guide:**
    1. **Upload** your `.xlsx` or `.xls` file.
    2. **Select** the sheet and the channels you wish to analyze from the sidebar.
    3. **Adjust** the signal processing parameters (Sampling Rate, RMS/Smoothing Windows) in the sidebar.
    4. **Click** the 'Process & Plot EMG Data' button to visualize the filtered signal and its smoothed RMS envelope.
    """)