import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.interpolate import interp1d
from datetime import datetime

# Page configuration
st.set_page_config(layout="wide",
                   page_title="Advanced SCADA Analysis",
                   page_icon="🌬️")
st.title("🌬️ Advanced SCADA Analysis - Vestas V90 2MW MK7")

# =============================================
# OFFICIAL VESTAS V90 2MW MK7 POWER CURVE DATA
# =============================================
power_curve_data = {
    'WS_m_s': [
        3.0, 3.5, 3.98, 4.51, 5.00, 5.50, 6.02, 6.49, 7.00, 7.49,
        8.01, 8.52, 9.00, 9.48, 10.01, 10.48, 10.98, 11.52, 12.01, 12.52,
        13.02, 13.47, 13.98, 14.52, 14.99, 15.49, 15.92, 16.43, 17.0, 17.50, 18.0, 19.0, 20.0, 21.0,
        22.0, 23.0, 24.0, 25.0
    ],
    'Production_KW': [
        0, 42.2, 93.3, 145.2, 211.3, 284.2, 390.9, 491.5, 601.1, 731.8,
        884.5, 1087.6, 1247.1, 1429.6, 1594.3, 1742.9, 1861.2, 1948.1,
        1993.3, 2003.5, 2007.0, 2007.7, 2007.3, 2007.4, 2007, 2006.8, 2006.7,
        2006.5, 2006.5, 2006.5, 2006.5, 2006.5, 2006.5, 2006.5, 2006.5, 2003.5, 1990, 0
    ]
}
if len(power_curve_data['WS_m_s']) != len(power_curve_data['Production_KW']):
    st.error("Wind speed and power arrays must have the same length")
    st.stop()

power_curve_df = pd.DataFrame(power_curve_data)
power_curve_df['Production_MW'] = power_curve_df['Production_KW'] / 1000

# Technical parameters
rated_power_default = 2.0
cut_in_default = 3.5
rated_start_default = 13.5
cut_out_default = 25.0
rotor_diameter = 90
swept_area = 6362
air_density_default = 1.225

# Power curve table
st.subheader("🔢 Official V90 2MW MK7 Power Curve Table")
st.dataframe(power_curve_df, height=350)

# Download button for Excel
excel_buffer_curve = BytesIO()
with pd.ExcelWriter(excel_buffer_curve, engine='xlsxwriter') as writer:
    power_curve_df.to_excel(writer, index=False, sheet_name='PowerCurve')
excel_buffer_curve.seek(0)
st.download_button(
    label="📥 Download power curve (Excel)",
    data=excel_buffer_curve.getvalue(),
    file_name="V90_2MW_MK7_power_curve.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Sidebar parameters
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    st.subheader("V90 2MW MK7 Turbine Parameters")
    st.markdown(
        f"""
        **Technical specifications:**
        - Rated power: {rated_power_default} MW
        - Rotor diameter: {rotor_diameter} m
        - Swept area: {swept_area} m²
        """
    )
    rated_power = st.number_input(
        "Rated Power (MW)", min_value=1.8, max_value=2.2,
        value=rated_power_default, step=0.1, disabled=True
    )
    cut_in = st.slider("Cut-in Speed (m/s)", 0.0, 5.0, cut_in_default, 0.1)
    rated_start = st.slider("Rated Speed (m/s)", 10.0, 15.0,
                            rated_start_default, 0.1)
    cut_out = st.slider("Cut-out Speed (m/s)", 20.0, 30.0,
                        cut_out_default, 0.5)

    st.subheader("Environmental Conditions")
    air_density = st.slider(
        "Air Density (kg/m³)", 1.0, 1.4, air_density_default, 0.01
    )
    density_correction = st.checkbox("Apply air density correction", value=True)

    st.subheader("Data Options")
    conversion_needed = st.selectbox(
        "Power Units",
        options=[
            "Data in MW (no conversion)",
            "Convert from kW",
            "Convert from W"
        ],
        index=0
    )

    st.subheader("Visualization Options")
    bin_size = st.slider("Bin Size (m/s)", 0.1, 1.0, 0.5, 0.1)
    transparency = st.slider("Point Transparency", 0.1, 1.0, 0.3, 0.1)

    st.subheader("Export Options")
    export_format = st.selectbox("Export Format", ["PNG", "PDF", "SVG"])

# Helper functions
def create_power_curve_interpolator():
    return interp1d(
        power_curve_df['WS_m_s'],
        power_curve_df['Production_MW'],
        kind='linear',
        bounds_error=False,
        fill_value=0
    )

def apply_density_correction(power_values, air_density):
    corrected = power_values * (air_density / 1.225)
    max_power = power_curve_df['Production_MW'].max()
    return np.minimum(corrected, max_power)

def save_figure(fig, filename):
    buf = BytesIO()
    if export_format == "PNG":
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    elif export_format == "PDF":
        fig.savefig(buf, format="pdf", bbox_inches='tight')
    elif export_format == "SVG":
        fig.savefig(buf, format="svg", bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# File upload
uploaded_file = st.file_uploader(
    "📤 Upload SCADA file (.xlsx or .csv)", type=["xlsx", "csv"]
)

if uploaded_file:
    try:
        # ---------- Read file ----------
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='latin1', decimal=',')
            sheet_name = "CSV Data"
        else:
            xls = pd.ExcelFile(uploaded_file)
            sheet = st.selectbox("Select sheet", xls.sheet_names,
                                 key="sheet_selector")
            df = pd.read_excel(uploaded_file, sheet_name=sheet)
            for col in df.select_dtypes(include='object'):
                df[col] = df[col].apply(
                    lambda x: str(x).replace(',', '.') if isinstance(x, str) else x
                )
            sheet_name = sheet

        st.subheader("📋 Complete Raw Data")
        st.dataframe(df, height=300)
        st.write(f"Total rows loaded: {len(df)} | Sheet: {sheet_name}")

        # ---------- Column detection ----------
        def find_column(frame, patterns):
            for col in frame.columns:
                if any(p.lower() in str(col).lower() for p in patterns):
                    return col
            return None

        cols_sel = st.columns(3)
        with cols_sel[0]:
            time_col = st.selectbox(
                "Time Column", df.columns,
                index=df.columns.get_loc(
                    find_column(df, ['date', 'time', 'timestamp'])
                ) if find_column(df, ['date', 'time', 'timestamp']) else 0
            )
        with cols_sel[1]:
            wind_col = st.selectbox(
                "Wind Speed Column", df.columns,
                index=df.columns.get_loc(
                    find_column(df, ['wind', 'ws', 'speed', 'velocity'])
                ) if find_column(df, ['wind', 'ws', 'speed', 'velocity']) else 0
            )
        with cols_sel[2]:
            power_col = st.selectbox(
                "Power Column", df.columns,
                index=df.columns.get_loc(
                    find_column(df, ['power', 'active', 'output'])
                ) if find_column(df, ['power', 'active', 'output']) else 0
            )

        # ---------- Data cleaning ----------
        df_clean = df.copy()
        try:
            df_clean['Timestamp'] = pd.to_datetime(
                df_clean[time_col],
                format='%d.%m.%Y %H:%M:%S',
                errors='coerce'
            )
            mask = df_clean['Timestamp'].isna()
            df_clean.loc[mask, 'Timestamp'] = pd.to_datetime(
                df_clean.loc[mask, time_col],
                format='%d.%m.%Y',
                errors='coerce'
            )
            mask = df_clean['Timestamp'].isna()
            df_clean.loc[mask, 'Timestamp'] = pd.to_datetime(
                df_clean.loc[mask, time_col],
                infer_datetime_format=True,
                errors='coerce'
            )
        except Exception:
            df_clean['Timestamp'] = pd.to_datetime(
                df_clean[time_col], errors='coerce'
            )

        df_clean['Wind_Raw'] = pd.to_numeric(
            df_clean[wind_col].astype(str).str.replace(',', '.'),
            errors='coerce'
        )
        df_clean['Power_Raw'] = pd.to_numeric(
            df_clean[power_col].astype(str).str.replace(',', '.'),
            errors='coerce'
        )

        df_clean['Was_Corrected'] = (
            (df_clean['Wind_Raw'] < 0) | (df_clean['Power_Raw'] < 0) |
            df_clean['Wind_Raw'].isna() | df_clean['Power_Raw'].isna()
        )
        df_clean['Wind_Raw'] = df_clean['Wind_Raw'].clip(lower=0).fillna(0)
        df_clean['Power_Raw'] = df_clean['Power_Raw'].clip(lower=0).fillna(0)

        if conversion_needed == "Convert from kW":
            df_clean['Power_Raw'] /= 1000
        elif conversion_needed == "Convert from W":
            df_clean['Power_Raw'] /= 1e6

        df_clean['Data_Quality'] = np.where(
            df_clean['Timestamp'].isna(), 'Invalid Data',
            np.where(df_clean['Was_Corrected'], 'Corrected to 0', 'Valid Data')
        )

        df_for_analysis = df_clean[df_clean['Data_Quality'] != 'Invalid Data']
        if df_for_analysis.empty:
            st.error("No valid data to analyse. Please check your file.")
            st.stop()

        # =============================================
        # FINAL STATISTICS
        # =============================================
        st.subheader("📊 Final Statistics")
        cols_stats = st.columns(4)
        cols_stats[0].metric("Total Data Points", len(df_clean))

        valid_cnt = len(df_for_analysis.query("Data_Quality == 'Valid Data'"))
        cols_stats[1].metric("Valid Data Points",
                             valid_cnt,
                             f"{valid_cnt / len(df_clean) * 100:.1f}%")

        # MODIFIED → now using all wind data from df_clean
        avg_wind = df_clean['Wind_Raw'].mean()
        avg_power = df_for_analysis['Power_Raw'].mean()
        cols_stats[2].metric(
            "Average Wind Speed",
            f"{avg_wind:.2f} m/s",
            f"±{df_clean['Wind_Raw'].std():.2f}"
        )
        cols_stats[3].metric(
            "Average Power",
            f"{avg_power:.3f} MW",
            f"±{df_for_analysis['Power_Raw'].std():.3f}"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("ℹ️ Please upload a SCADA data file to begin analysis")
