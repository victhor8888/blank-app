import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
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
        13.02, 13.47, 13.98, 14.52, 14.99, 15.49, 15.92, 16.43, 17.0, 18.0, 19.0, 20.0, 21.0,
        22.0, 23.0, 24.0, 25.0
    ],
    'Production_KW': [
        0, 42.2, 93.3, 145.2, 211.3, 284.2, 390.9, 491.5, 601.1, 731.8,
        884.5, 1087.6, 1247.1, 1429.6, 1594.3, 1742.9, 1861.2, 1948.1,
        1993.3, 2003.5, 2007.0, 2007.7, 2007.3, 2007.4, 2007, 2006.8, 2006.7,
        2006.5, 2006.5, 2006.5, 2006.5, 2006.5, 2006.5, 2006.5, 2003.5, 1990, 0
    ]
}
# Verificación mejorada de longitud
if len(power_curve_data['WS_m_s']) != len(power_curve_data['Production_KW']):
    st.error("Wind speed and power arrays must have the same length")
    st.stop()

power_curve_df = pd.DataFrame(power_curve_data)
power_curve_df['Production_MW'] = power_curve_df['Production_KW'] / 1000

# Official V90 2MW MK7 technical parameters
rated_power_default = 2.0          # MW
cut_in_default = 3.5               # m/s
rated_start_default = 13.5         # m/s
cut_out_default = 25.0             # m/s
rotor_diameter = 90                # m
swept_area = 6362                  # m²
air_density_default = 1.225        # kg/m³

# =============================================
# DYNAMIC POWER CURVE TABLE
# =============================================
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

# =============================================
# SIDEBAR PARAMETERS (UPDATED WITH OFFICIAL VALUES)
# =============================================
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

# =============================================
# MAIN FUNCTIONS
# =============================================
def create_power_curve_interpolator():
    """Creates interpolator for the provided power curve"""
    return interp1d(
        power_curve_df['WS_m_s'],
        power_curve_df['Production_MW'],
        kind='linear',
        bounds_error=False,
        fill_value=0
    )

def apply_density_correction(power_values, air_density):
    """Applies air density correction"""
    corrected = power_values * (air_density / 1.225)
    max_power = power_curve_df['Production_MW'].max()
    return np.minimum(corrected, max_power)

def save_figure(fig, filename):
    """Saves figure in selected format"""
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

# =============================================
# FILE PROCESSING WITH DATA CLEANING
# =============================================
uploaded_file = st.file_uploader(
    "📤 Upload SCADA file (.xlsx or .csv)", type=["xlsx", "csv"]
)

if uploaded_file:
    try:
        # ---------- Read file (handles European formats) ----------
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

        # ---------- Helper to auto-detect columns ----------
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

        # ---------- Cleaning ----------
        df_clean = df.copy()
        # Robust European date handling
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

        # Numeric conversion (comma → dot)
        df_clean['Wind_Raw'] = pd.to_numeric(
            df_clean[wind_col].astype(str).str.replace(',', '.'),
            errors='coerce'
        )
        df_clean['Power_Raw'] = pd.to_numeric(
            df_clean[power_col].astype(str).str.replace(',', '.'),
            errors='coerce'
        )

        # Basic sanitisation
        df_clean['Was_Corrected'] = (
            (df_clean['Wind_Raw'] < 0) | (df_clean['Power_Raw'] < 0) |
            df_clean['Wind_Raw'].isna() | df_clean['Power_Raw'].isna()
        )
        df_clean['Wind_Raw'] = df_clean['Wind_Raw'].clip(lower=0).fillna(0)
        df_clean['Power_Raw'] = df_clean['Power_Raw'].clip(lower=0).fillna(0)

        # Unit conversion
        if conversion_needed == "Convert from kW":
            df_clean['Power_Raw'] /= 1000
        elif conversion_needed == "Convert from W":
            df_clean['Power_Raw'] /= 1e6

        # Data classification
        df_clean['Data_Quality'] = np.where(
            df_clean['Timestamp'].isna(), 'Invalid Data',
            np.where(df_clean['Was_Corrected'], 'Corrected to 0', 'Valid Data')
        )

        df_for_analysis = df_clean[df_clean['Data_Quality'] != 'Invalid Data']
        if df_for_analysis.empty:
            st.error("No valid data to analyse. Please check your file.")
            st.stop()

        # =============================================
        # VISUALIZATIONS
        # =============================================
        plt.style.use('seaborn-v0_8')
        colors = {'Valid Data': 'green',
                  'Corrected to 0': 'red',
                  'Invalid Data': 'gray'}

        # 1. Data quality distribution
        st.subheader("📊 Data Quality Distribution")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        df_clean['Data_Quality'].value_counts().plot(
            kind='pie',
            autopct='%1.1f%%',
            colors=[colors[k] for k in df_clean['Data_Quality'].unique()],
            ax=ax1
        )
        ax1.set_ylabel('')
        ax1.set_title(f"Data Quality – {sheet_name}\nV90 2MW MK7 Turbine")
        st.pyplot(fig1)
        st.download_button(
            label=f"⬇️ Download Quality Chart ({export_format})",
            data=save_figure(fig1, "data_quality"),
            file_name=f"data_quality_{sheet_name}.{export_format.lower()}",
            mime="application/pdf" if export_format == "PDF"
                 else f"image/{export_format.lower()}"
        )

        # 2. Power time series
        st.subheader("📈 Power Time Series (All Data)")
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        for quality, group in df_clean.groupby('Data_Quality'):
            ax2.scatter(
                group['Timestamp'], group['Power_Raw'],
                color=colors.get(quality, 'blue'),
                alpha=0.5, s=10
            )
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Power (MW)')
        ax2.grid(alpha=0.3)
        ax2.set_title(f"Time Series – {sheet_name}\nV90 2MW MK7 Turbine")
        st.pyplot(fig2)
        st.download_button(
            label=f"⬇️ Download Time Series ({export_format})",
            data=save_figure(fig2, "time_series"),
            file_name=f"time_series_{sheet_name}.{export_format.lower()}",
            mime="application/pdf" if export_format == "PDF"
                 else f"image/{export_format.lower()}"
        )

        # 3. Power curve (all data)
        st.subheader("🌪️ Power Curve (All Data)")
        fig3, ax3 = plt.subplots(figsize=(12, 7))
        for quality, group in df_clean.groupby('Data_Quality'):
            ax3.scatter(
                group['Wind_Raw'], group['Power_Raw'],
                color=colors.get(quality, 'blue'),
                alpha=transparency, s=15
            )
        wind_range = np.linspace(0, cut_out, 100)
        power_reference = create_power_curve_interpolator()(wind_range)
        if density_correction:
            power_reference = apply_density_correction(power_reference,
                                                       air_density)
        ax3.plot(wind_range, power_reference, 'r-', lw=3,
                 label='Official V90 Curve')
        ax3.scatter(
            power_curve_df['WS_m_s'], power_curve_df['Production_MW'],
            color='blue', s=50, label='Official Points'
        )
        ax3.axvline(cut_in, color='gray', ls='--', alpha=0.5,
                    label=f'Cut-in ({cut_in} m/s)')
        ax3.axvline(rated_start, color='green', ls='--', alpha=0.5,
                    label=f'Rated ({rated_start} m/s)')
        ax3.axvline(cut_out, color='red', ls='--', alpha=0.5,
                    label=f'Cut-out ({cut_out} m/s)')
        ax3.axhline(rated_power, color='purple', ls='--', alpha=0.5,
                    label=f'Rated Power ({rated_power} MW)')
        ax3.set_xlabel('Wind Speed (m/s)')
        ax3.set_ylabel('Power (MW)')
        ax3.grid(alpha=0.3)
        ax3.legend()
        ax3.set_title(
            f"Power Curve – {sheet_name}\nV90 2MW MK7 Turbine "
            f"(Air density: {air_density} kg/m³)"
        )
        st.pyplot(fig3)
        st.download_button(
            label=f"⬇️ Download Power Curve ({export_format})",
            data=save_figure(fig3, "power_curve"),
            file_name=f"power_curve_{sheet_name}.{export_format.lower()}",
            mime="application/pdf" if export_format == "PDF"
                 else f"image/{export_format.lower()}"
        )

        # =============================================
        # VALID DATA ANALYSIS
        # =============================================
        if not df_for_analysis.empty:
            st.subheader("📌 Valid Data Analysis")

            # 4. Power curve (valid data only)
            fig4, ax4 = plt.subplots(figsize=(12, 7))
            valid_data = df_for_analysis.query("Data_Quality == 'Valid Data'")
            sc = ax4.scatter(
                valid_data['Wind_Raw'], valid_data['Power_Raw'],
                c=(valid_data['Timestamp'].astype('int64') // 1e9),
                cmap='viridis', alpha=transparency, s=15
            )
            corrected = df_for_analysis.query("Data_Quality == 'Corrected to 0'")
            ax4.scatter(
                corrected['Wind_Raw'], corrected['Power_Raw'],
                color='red', alpha=transparency, s=15
            )

            bins = np.arange(
                0, df_for_analysis['Wind_Raw'].max() + bin_size, bin_size
            )
            df_for_analysis['Wind_bin'] = pd.cut(
                df_for_analysis['Wind_Raw'], bins
            )
            binned = (df_for_analysis
                      .groupby('Wind_bin')
                      .agg(Wind_mean=('Wind_Raw', 'mean'),
                           Power_mean=('Power_Raw', 'mean'),
                           Power_std=('Power_Raw', 'std'),
                           Count=('Power_Raw', 'count'))
                      .reset_index()
                      .query('Count > 0'))

            ax4.plot(wind_range, power_reference, 'r-', lw=3,
                     label='Official V90 Curve')
            ax4.scatter(
                power_curve_df['WS_m_s'], power_curve_df['Production_MW'],
                color='blue', s=50, label='Official Points'
            )
            
            # SOLO UNA LÍNEA PARA LOS PROMEDIOS (SIN BARRAS DE ERROR)
            ax4.plot(binned['Wind_mean'], binned['Power_mean'], 'k-', label='Binned Averages')
            
            ax4.axvline(cut_in, color='gray', ls='--', alpha=0.5)
            ax4.axvline(rated_start, color='green', ls='--', alpha=0.5)
            ax4.axvline(cut_out, color='red', ls='--', alpha=0.5)
            ax4.axhline(rated_power, color='purple', ls='--', alpha=0.5)

            ax4.set_xlabel('Wind Speed (m/s)')
            ax4.set_ylabel('Power (MW)')
            ax4.grid(alpha=0.3)
            ax4.legend()
            ax4.set_title(
                f"Power Curve (Valid Data) – {sheet_name}\nV90 2MW MK7 Turbine"
            )
            cbar = plt.colorbar(sc, ax=ax4)
            cbar.set_label('Timestamp')
            st.pyplot(fig4)
            st.download_button(
                label=f"⬇️ Download Valid Data Curve ({export_format})",
                data=save_figure(fig4, "power_curve_valid"),
                file_name=f"valid_curve_{sheet_name}.{export_format.lower()}",
                mime="application.pdf" if export_format == "PDF"
                     else f"image/{export_format.lower()}"
            )

            # 5. Daily efficiency
            st.subheader("📅 Daily Efficiency (Valid Data)")
            if not valid_data.empty:
                valid_data['Expected_Power'] = valid_data['Wind_Raw'].apply(
                    lambda w: float(create_power_curve_interpolator()(w))
                )
                valid_data['Efficiency'] = (
                    valid_data['Power_Raw'] /
                    valid_data['Expected_Power'].replace(0, np.nan) * 100
                )
                daily_eff = valid_data.set_index('Timestamp')['Efficiency'] \
                                       .resample('D').mean()

                fig5, ax5 = plt.subplots(figsize=(14, 5))
                ax5.plot(daily_eff.index, daily_eff.values,
                         'b-o', ms=4, lw=1)
                ax5.axhline(100, color='r', ls='--', alpha=0.5)
                ax5.fill_between(
                    daily_eff.index, daily_eff.values, 100,
                    where=(daily_eff.values < 100),
                    color='red', alpha=0.1, label='Below Expected'
                )
                ax5.fill_between(
                    daily_eff.index, daily_eff.values, 100,
                    where=(daily_eff.values >= 100),
                    color='green', alpha=0.1, label='Above Expected'
                )
                ax5.set_xlabel('Date')
                ax5.set_ylabel('Efficiency (%)')
                ax5.grid(alpha=0.3)
                ax5.legend()
                ax5.set_title(
                    f"Daily Efficiency (Valid Data) – {sheet_name}\n"
                    "V90 2MW MK7 Turbine"
                )
                st.pyplot(fig5)
                st.download_button(
                    label=f"⬇️ Download Daily Efficiency ({export_format})",
                    data=save_figure(fig5, "daily_efficiency"),
                    file_name=f"daily_efficiency_{sheet_name}.{export_format.lower()}",
                    mime="application.pdf" if export_format == "PDF"
                         else f"image/{export_format.lower()}"
                )

        # =============================================
        # DATA EXPORT
        # =============================================
        st.subheader("💾 Data Export")
        export_df = df_clean[['Timestamp', 'Wind_Raw',
                              'Power_Raw',
                              'Data_Quality', 'Was_Corrected']].copy()
        export_df['Timestamp'] = export_df['Timestamp'] \
            .dt.strftime('%Y-%m-%d %H:%M:%S')

        if 'Expected_Power' in locals():
            export_df['Expected_Power'] = export_df['Expected_Power']
            export_df['Efficiency'] = export_df['Efficiency']

        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, sheet_name='Processed_Data',
                               index=False)
            if 'binned' in locals():
                binned.to_excel(writer, sheet_name='Bin_Averages', index=False)

        st.download_button(
            label="📥 Download Processed Data (Excel)",
            data=excel_buffer.getvalue(),
            file_name=f"SCADA_analysis_{sheet_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

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

        avg_wind = df_for_analysis['Wind_Raw'].mean()
        avg_power = df_for_analysis['Power_Raw'].mean()
        cols_stats[2].metric(
            "Average Wind Speed",
            f"{avg_wind:.2f} m/s",
            f"±{df_for_analysis['Wind_Raw'].std():.2f}"
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
