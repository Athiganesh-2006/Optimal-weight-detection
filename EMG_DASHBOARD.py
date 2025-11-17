# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import butter, sosfiltfilt
from typing import Optional, Tuple

st.set_page_config(page_title="EMG Dashboard (Streamlit)", layout="wide")

# ---------- Helper functions ----------
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
    sum_sq = np.convolve(sq, kernel, mode='same')
    # count of valid elements per window
    valid = (~np.isnan(arr)).astype(float)
    count = np.convolve(valid, kernel, mode='same')
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
    sum_vals = np.convolve(arr_n, kernel, mode='same')
    valid = (~np.isnan(arr)).astype(float)
    count = np.convolve(valid, np.ones(window_samples), mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        ma = sum_vals / count
    ma[count == 0] = np.nan
    return ma

def design_filter(filter_type: str, cutoff: Tuple[float, Optional[float]], fs: float, order=4):
    """
    Return second-order-sections (sos) for butterworth filter.
    filter_type: 'low', 'high', 'band'
    cutoff: for 'low' or 'high' single value (float,)
            for 'band' tuple (lowcut, highcut)
    fs: sampling freq
    """
    ny = 0.5 * fs
    if filter_type == 'band':
        low, high = cutoff
        if low <= 0 or high <= 0 or high <= low:
            return None
        lown = low / ny
        highn = high / ny
        sos = butter(order, [lown := lown, highn], btype='bandpass', output='sos')
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
    # handle NaNs by linear interpolation for filter continuity then reapply NaN mask
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

# ---------- UI ----------
st.title("EMG Dashboard — Streamlit")
st.markdown("Upload an Excel workbook (.xlsx/.xls) containing EMG channels across sheets. "
            "First row should be headers; first column is assumed to be X/time.")

uploaded_file = st.file_uploader("Choose Excel file", type=['xlsx','xls'])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    try:
        sheets = read_workbook(file_bytes)
    except Exception as e:
        st.error(f"Failed to read workbook: {e}")
        st.stop()

    # Sidebar: sheet selection + processing params
    st.sidebar.header("Sheet & Channel Selection")
    sheet_names = list(sheets.keys())
    sel_sheet = st.sidebar.selectbox("Sheet", sheet_names)

    df = sheets[sel_sheet].copy()
    st.subheader(f"Sheet: {sel_sheet}")
    st.write("Preview (first 10 rows):")
    st.dataframe(df.head(10))

    # columns and x-axis
    cols = df.columns.tolist()
    if len(cols) < 2:
        st.warning("Sheet must have at least two columns (X + one channel).")
        st.stop()

    x_col = cols[0]
    st.sidebar.write(f"Assuming X-axis column (time): **{x_col}**")
    # coerce X to numeric if possible, otherwise keep as label
    x_series = to_numeric_series(df[x_col])
    x_is_numeric = not x_series.isna().all()

    channel_choices = cols[1:]
    sel_channels = st.sidebar.multiselect("Channels to plot", channel_choices, default=channel_choices[:min(3,len(channel_choices))])

    st.sidebar.header("Processing Parameters")
    sample_rate = st.sidebar.number_input("Sample rate (Hz)", value=2000.0, min_value=1.0, format="%.1f")
    rms_window_ms = st.sidebar.number_input("RMS window (ms)", value=200.0, min_value=1.0)
    smooth_window_samples = st.sidebar.number_input("Smoothing window (samples)", value=5, min_value=1, step=1)

    # filter options
    st.sidebar.subheader("Filters (Butterworth)")
    hp_cut = st.sidebar.number_input("Highpass cutoff (Hz) — 0 to disable", value=20.0, min_value=0.0)
    lp_cut = st.sidebar.number_input("Lowpass cutoff (Hz) — 0 to disable", value=450.0, min_value=0.0)
    # option for bandpass overrides hp/lp if both provided? We'll implement HP then LP (cascade),
    # but also provide bandpass option (checkbox)
    bandpass_enabled = st.sidebar.checkbox("Use bandpass (low & high together)", value=False)
    if bandpass_enabled:
        bp_low = st.sidebar.number_input("Bandpass low cutoff (Hz)", value=20.0, min_value=0.0)
        bp_high = st.sidebar.number_input("Bandpass high cutoff (Hz)", value=450.0, min_value=0.0)

    # action button
    if st.sidebar.button("Process & Plot"):

        # prepare x values
        if x_is_numeric:
            x_vals = x_series.to_numpy(dtype=float)
        else:
            # if X is non-numeric keep index as x
            x_vals = np.arange(len(df))

        # create figure
        fig = go.Figure()
        processed_tables = {}  # store for CSV export

        # compute windows
        rms_window_samples = max(1, int(round((rms_window_ms / 1000.0) * sample_rate)))
        smooth_window_samples = max(1, int(smooth_window_samples))

        # design filters
        sos_hp = design_filter('high', hp_cut, sample_rate, order=4) if (not bandpass_enabled and hp_cut > 0) else None
        sos_lp = design_filter('low', lp_cut, sample_rate, order=4) if (not bandpass_enabled and lp_cut > 0) else None
        sos_bp = None
        if bandpass_enabled:
            sos_bp = design_filter('band', (bp_low, bp_high), sample_rate, order=4)

        for ch in sel_channels:
            series = to_numeric_series(df[ch])
            arr = series.to_numpy(dtype=float)

            # apply bandpass or cascade hp->lp
            if bandpass_enabled and sos_bp is not None:
                proc = apply_sos_filter(arr, sos_bp)
            else:
                proc = arr.copy()
                if sos_hp is not None:
                    proc = apply_sos_filter(proc, sos_hp)
                if sos_lp is not None:
                    proc = apply_sos_filter(proc, sos_lp)

            # RMS
            rms = moving_rms(proc, rms_window_samples)
            # smoothing
            rms_smooth = moving_average(rms, smooth_window_samples)

            # plot: raw, rms, rms_smooth
            # use different trace styles
            fig.add_trace(go.Scatter(x=x_vals, y=proc, mode='lines', name=f"{ch} (raw)", line=dict(width=1)))
            fig.add_trace(go.Scatter(x=x_vals, y=rms, mode='lines', name=f"{ch} (RMS)", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=x_vals, y=rms_smooth, mode='lines', name=f"{ch} (RMS-smoothed)", line=dict(dash='dash')))

            # table for export
            table_df = pd.DataFrame({x_col: x_vals, f"{ch}_raw": proc, f"{ch}_rms": rms, f"{ch}_rms_smoothed": rms_smooth})
            processed_tables[ch] = table_df

        fig.update_layout(height=600, template="plotly_white",
                          title=f"Sheet: {sel_sheet} — Channels: {', '.join(sel_channels)}",
                          xaxis_title=x_col, yaxis_title="Value",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))

        st.plotly_chart(fig, use_container_width=True)

        # create combined CSV (X + columns for each selected channel)
        # We'll merge processed tables on index (which corresponds to row)
        combined = None
        for ch, tdf in processed_tables.items():
            if combined is None:
                combined = tdf
            else:
                # drop duplicate X to avoid multiple columns with same name
                combined = pd.concat([combined, tdf.drop(columns=[x_col])], axis=1)

        if combined is not None:
            csv_bytes = combined.to_csv(index=False).encode('utf-8')
            st.download_button("Download processed CSV", data=csv_bytes, file_name=f"{sel_sheet}_processed.csv", mime="text/csv")

            # try to offer PNG download via plotly to_image (kaleido required)
            try:
                png_bytes = fig.to_image(format="png", width=1600, height=900, scale=1)
                st.download_button("Download chart PNG", data=png_bytes, file_name=f"{sel_sheet}_chart.png", mime="image/png")
            except Exception as ex:
                st.info("PNG export requires 'kaleido'. Install with `pip install kaleido` to enable chart PNG download.")
                st.write(f"(PNG export error: {ex})")

        # show some notes
        st.markdown("**Notes**")
        st.write(f"RMS window: {rms_window_ms} ms ≈ {rms_window_samples} samples at {sample_rate} Hz. Smoothing window: {smooth_window_samples} samples.")
        if bandpass_enabled:
            st.write(f"Applied bandpass: {bp_low} Hz – {bp_high} Hz (Butterworth, order=4).")
        else:
            if hp_cut > 0:
                st.write(f"Applied highpass: {hp_cut} Hz (Butterworth, order=4).")
            if lp_cut > 0:
                st.write(f"Applied lowpass: {lp_cut} Hz (Butterworth, order=4).")

else:
    st.info("Upload an Excel file to begin. Example: first row headers, first column time in seconds or sample index.")
    st.markdown("""
    **Tip**
    - If your data's first column is non-numeric (labels), the app will use row index as X-axis.
    - For best filter results, ensure `sample_rate` matches how the time axis was sampled.
    - To export PNG of the Plotly figure, install `kaleido`: `pip install kaleido`.
    """)
