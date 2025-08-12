import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
import datetime

# =========================
# Official power curve
# =========================
power_curve_data = {
    'WS_m_s': [
        3.0, 3.5, 3.98, 4.51, 5.00, 5.50, 6.02, 6.49, 7.00, 7.49,
        8.01, 8.52, 9.00, 9.48, 10.01, 10.48, 10.98, 11.52, 12.01,
        12.52, 13.02, 13.47, 13.98, 14.52, 14.99, 15.49, 15.92,
        16.43, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0
    ],
    'Production_KW': [
        0, 42.2, 93.3, 145.2, 211.3, 284.2, 390.9, 491.5, 601.1,
        731.8, 884.5, 1087.6, 1247.1, 1429.6, 1594.3, 1742.9, 1861.2,
        1948.1, 1993.3, 2003.5, 2007.0, 2007.7, 2007.3, 2007.4, 2007.0,
        2006.8, 2006.7, 2006.5, 2006.5, 2006.5, 2006.5, 2006.5, 2006.5,
        2006.5, 2003.5, 1990, 0
    ]
}

if len(power_curve_data['WS_m_s']) != len(power_curve_data['Production_KW']):
    st.error("Power curve data arrays have mismatched lengths.")
    st.stop()

power_curve_df = pd.DataFrame(power_curve_data)
power_curve_df['Production_MW'] = power_curve_df['Production_KW'] / 1000

# =========================
# Default parameters
# =========================
rated_power_default = 2.0  # MW
cut_in_default = 3.5  # m/s
rated_start_default = 13.5  # m/s
cut_out_default = 25.0  # m/s
rotor_diameter = 90  # m
swept_area = 6362  # m^2
air_density_default = 1.225  # kg/m3 (standard)

# =========================
# Display power curve
# =========================
st.set_page_config(layout="wide", page_title="ERS Advanced SCADA Analysis", page_icon="ERS")
st.title("ERS Advanced SCADA Analysis - Vestas V90 2MW MK7")
st.subheader("ERS - Official V90 2MW MK7 Power Curve Table")
st.dataframe(power_curve_df, height=350)
excel_buffer_curve = BytesIO()
with pd.ExcelWriter(excel_buffer_curve, engine='xlsxwriter') as writer:
    power_curve_df.to_excel(writer, index=False, sheet_name='PowerCurve')
excel_buffer_curve.seek(0)
st.download_button(
    label="💾 Download power curve (Excel)",
    data=excel_buffer_curve.getvalue(),
    file_name="V90_2MW_MK7_power_curve.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# =========================
# Sidebar parameters
# =========================
with st.sidebar:
    st.header("🛠️ Analysis Settings")
    st.subheader("V90 2MW MK7 Turbine Parameters")
    st.markdown(
        f"""**Technical specifications:**
- Rated power: {rated_power_default} MW
- Rotor diameter: {rotor_diameter} m
- Swept area: {swept_area} m²
""")
    rated_power = st.number_input("Rated Power (MW)", min_value=1.8, max_value=2.2,
                                  value=rated_power_default, step=0.1, disabled=True)
    cut_in = st.slider("Cut-in Speed (m/s)", 0.0, 5.0, cut_in_default, 0.1)
    rated_start = st.slider("Rated Speed (m/s)", 10.0, 15.0, rated_start_default, 0.1)
    cut_out = st.slider("Cut-out Speed (m/s)", 20.0, 30.0, cut_out_default, 0.5)

    st.subheader("Environmental Conditions")
    air_density = st.slider("Air Density (kg/m³)", 1.0, 1.4, air_density_default, 0.01)
    density_correction = st.checkbox("Apply air density correction", value=True)

    st.subheader("Data Options")
    conversion_needed = st.selectbox("Power Units",
                                    ["Data in MW (no conversion)", "Convert from kW", "Convert from W"], index=0)

    st.subheader("Data Sampling")
    sample_fraction = st.slider("Sample fraction (for large files)", 0.1, 1.0, 1.0, 0.1)
    
    st.subheader("🗓️ Filter by Date Range")
    today = datetime.date.today()
    start_date = st.date_input('Start date', value=today - datetime.timedelta(days=30))
    end_date = st.date_input('End date', value=today)

    st.subheader("Visualization Options")
    bin_size = st.slider("Bin Size (m/s)", 0.1, 1.0, 0.5, 0.1)
    transparency = st.slider("Point Transparency", 0.1, 1.0, 0.3, 0.1)

    st.subheader("Export Options")
    export_format = st.selectbox("Export Format", ["PNG", "PDF", "SVG"])

# =========================
# Helper functions
# =========================
def get_expected_power(wind_speed_series):
    return np.interp(
        wind_speed_series,
        power_curve_df['WS_m_s'],
        power_curve_df['Production_MW'],
        left=0,
        right=0
    )

def apply_density_correction(power_values, air_density):
    corrected = power_values * (air_density / 1.225)
    max_power = power_curve_df['Production_MW'].max()
    return np.minimum(corrected, max_power)

def set_sensible_limits(fig, df):
    if df.empty:
        return
    fig.update_xaxes(range=[0, 30])
    fig.update_yaxes(range=[0, rated_power * 1.1])

def find_column(frame, patterns):
    for col in frame.columns:
        col_lower = str(col).lower()
        if any(p.lower() in col_lower for p in patterns):
            return col
    return None
    
def process_turbine_data(uploaded_file, conversion_needed, sample_fraction, start_date, end_date):
    if uploaded_file is None:
        return None, None
    
    sheet_name = None
    
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(uploaded_file, encoding='latin1')
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                return None, None
        else:
            try:
                xls = pd.ExcelFile(uploaded_file)
                if not xls.sheet_names:
                    st.error("Excel file contains no sheets")
                    return None, None
                sheet_name = st.selectbox(f"Select sheet for {uploaded_file.name}", xls.sheet_names)
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                return None, None

        time_col_raw = find_column(df, ['date', 'time', 'timestamp'])
        if time_col_raw:
            df['Timestamp'] = pd.to_datetime(df[time_col_raw], errors='coerce')
            df = df.loc[(df['Timestamp'] >= pd.to_datetime(start_date)) & (df['Timestamp'] <= pd.to_datetime(end_date))]
            if df.empty:
                st.warning(f"No data found for {uploaded_file.name} for the selected date range.")
                return None, None
        
        if sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=42)

        time_col_selected = find_column(df, ['date', 'time', 'timestamp'])
        wind_col_selected = find_column(df, ['wind', 'ws', 'speed', 'velocity'])
        power_col_selected = find_column(df, ['power', 'active', 'output'])

        if not wind_col_selected or not power_col_selected:
            st.error(f"Could not automatically find wind and power columns in {uploaded_file.name}.")
            return None, None
        
        df_clean = df.copy()
        
        for col in [wind_col_selected, power_col_selected]:
            df_clean[col] = df_clean[col].astype(str).str.replace(',', '.', regex=False)
        
        try:
            df_clean['Timestamp'] = pd.to_datetime(df_clean[time_col_selected], errors='coerce')
            if df_clean['Timestamp'].isna().all():
                st.warning("Could not parse any dates in selected column. Using index as timestamp.")
                df_clean['Timestamp'] = pd.to_datetime(df_clean.index)
        except Exception as e:
            st.warning(f"Failed to parse time column: {str(e)}. Using index as timestamp.")
            df_clean['Timestamp'] = pd.to_datetime(df_clean.index)

        df_clean['Wind_Raw'] = pd.to_numeric(df_clean[wind_col_selected], errors='coerce')
        df_clean['Power_Raw'] = pd.to_numeric(df_clean[power_col_selected], errors='coerce')

        df_clean['is_invalid_time'] = df_clean['Timestamp'].isna()
        df_clean['is_invalid_value'] = (
            df_clean['Wind_Raw'].isna() | (df_clean['Wind_Raw'] < 0) |
            df_clean['Power_Raw'].isna() | (df_clean['Power_Raw'] < 0)
        )

        df_clean['Data_Quality'] = 'Valid Data'
        df_clean.loc[df_clean['is_invalid_time'], 'Data_Quality'] = 'Invalid Data'
        df_clean.loc[df_clean['is_invalid_value'] & ~df_clean['is_invalid_time'], 'Data_Quality'] = 'Corrected to 0'

        df_clean['Wind_Raw'] = df_clean['Wind_Raw'].clip(lower=0).fillna(0)
        df_clean['Power_Raw'] = df_clean['Power_Raw'].clip(lower=0).fillna(0)

        if conversion_needed == "Convert from kW":
            df_clean['Power_Raw'] /= 1000
        elif conversion_needed == "Convert from W":
            df_clean['Power_Raw'] /= 1e6
        
        return df_clean, sheet_name

    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        return None, None

# =========================
# File upload and processing
# =========================
uploaded_file = st.file_uploader("📄 Upload SCADA file for Turbine 1 (.xlsx or .csv)", type=["xlsx", "csv"])

st.sidebar.subheader("🗄️ Database Integration (Optional)")
db_type = st.sidebar.selectbox("Select Database Type", ["None", "PostgreSQL", "MySQL", "SQL Server"])
if db_type != "None":
    st.sidebar.warning("This feature is for demonstration. Actual database connection details are required.")
    db_host = st.sidebar.text_input("Host")
    db_user = st.sidebar.text_input("User")
    db_password = st.sidebar.text_input("Password", type="password")
    db_name = st.sidebar.text_input("Database Name")
    query = st.sidebar.text_area("SQL Query")
    if st.sidebar.button("Fetch Data from DB"):
        st.info("Simulating data retrieval from database...")
        st.success("Data fetched successfully (Simulated)")
        db_data = {'timestamp': pd.to_datetime(pd.date_range('2023-01-01', periods=100, freq='H')),
                   'wind_speed': np.random.uniform(3, 25, 100),
                   'power': np.random.uniform(0, 2000, 100)}
        df = pd.DataFrame(db_data)
        st.subheader("Data from Database")
        st.dataframe(df)

st.sidebar.subheader("📈 Turbine Comparison")
compare_mode = st.sidebar.checkbox("Compare with another turbine?", value=False)
uploaded_file_2 = None
if compare_mode:
    uploaded_file_2 = st.sidebar.file_uploader("📄 Upload SCADA file for Turbine 2", type=["xlsx", "csv"])

if uploaded_file:
    df_clean_1, sheet_name_1 = process_turbine_data(uploaded_file, conversion_needed, sample_fraction, start_date, end_date)
    
    if df_clean_1 is None:
        st.error("Could not process Turbine 1 file. Please check file format and content.")
        st.stop()
    
    df_for_analysis_1 = df_clean_1.copy()
    
    sheet_suffix_1 = f" - {sheet_name_1}" if sheet_name_1 else ""
    
    st.subheader(f"📋 Raw Data ({uploaded_file.name}{sheet_suffix_1})")
    st.dataframe(df_clean_1, height=300)
    st.write(f"Total rows loaded: {len(df_clean_1)}")
    
    df_for_analysis = df_for_analysis_1.copy()
    df_for_analysis['Expected_Power'] = get_expected_power(df_for_analysis['Wind_Raw'])
    if density_correction:
        df_for_analysis['Expected_Power'] = apply_density_correction(df_for_analysis['Expected_Power'], air_density)
    
    df_for_analysis['Efficiency'] = np.where(
        df_for_analysis['Expected_Power'] > 0,
        (df_for_analysis['Power_Raw'] / df_for_analysis['Expected_Power']) * 100,
        0
    )
    
    st.subheader("🌀 Turbulence Analysis")
    if not df_for_analysis.empty:
        df_for_analysis['Wind_bin'] = pd.cut(df_for_analysis['Wind_Raw'], bins=np.arange(0, 30, bin_size))
        binned_turbulence = (
            df_for_analysis.groupby('Wind_bin')
                .agg(Wind_mean=('Wind_Raw', 'mean'),
                     Wind_std=('Wind_Raw', 'std'),
                     Power_mean=('Power_Raw', 'mean'),
                     Count=('Wind_Raw', 'count'))
                .reset_index()
                .query('Count > 10')
        )
        binned_turbulence['Turbulence_Intensity'] = (binned_turbulence['Wind_std'] / binned_turbulence['Wind_mean']) * 100
        
        fig_turbulence = go.Figure()
        fig_turbulence.add_trace(go.Scatter(
            x=binned_turbulence['Wind_mean'], 
            y=binned_turbulence['Turbulence_Intensity'],
            mode='markers+lines',
            name='Turbulence Intensity',
            marker=dict(color='orange', size=8)
        ))
        fig_turbulence.update_layout(
    title=f"Turbulence Intensity vs. Wind Speed (Turbine 1{sheet_suffix_1})",
    xaxis_title="Average Wind Speed (m/s)",
    yaxis_title="Turbulence Intensity (%)",
    title_x=0.3,
    title_font=dict(size=18),
    height=500
)
        st.plotly_chart(fig_turbulence, use_container_width=True)
    else:
        st.info("No valid data to perform turbulence analysis.")
    
    st.subheader("🕵️ Anomaly Detection")
    if not df_for_analysis.empty:
        df_for_analysis['Power_Deviation'] = df_for_analysis['Power_Raw'] - df_for_analysis['Expected_Power']
        anomaly_threshold = -0.2
        df_for_analysis['is_anomaly'] = (
            (df_for_analysis['Power_Deviation'] / df_for_analysis['Expected_Power'] < anomaly_threshold) &
            (df_for_analysis['Wind_Raw'] > 5) & 
            (df_for_analysis['Wind_Raw'] < 20)
        )
        
        anomalies = df_for_analysis[df_for_analysis['is_anomaly']]
        if not anomalies.empty:
            st.warning(f"🚨 Found {len(anomalies)} potential anomalies.")
            with st.expander("View Anomaly Details"):
                st.dataframe(anomalies[['Timestamp', 'Wind_Raw', 'Power_Raw', 'Expected_Power', 'Power_Deviation']])
            
            fig_anomalies = go.Figure()
            fig_anomalies.add_trace(go.Scatter(
                x=df_for_analysis['Wind_Raw'], y=df_for_analysis['Power_Raw'],
                mode='markers',
                name='All Data',
                marker=dict(color='gray', opacity=0.3, size=5)
            ))
            fig_anomalies.add_trace(go.Scatter(
                x=anomalies['Wind_Raw'], y=anomalies['Power_Raw'],
                mode='markers',
                name='Detected Anomalies',
                marker=dict(color='red', size=8, symbol='x')
            ))
            fig_anomalies.update_layout(
    title=f"Anomaly Detection on Power Curve (Turbine 1{sheet_suffix_1})",
    xaxis_title="Wind Speed (m/s)",
    yaxis_title="Power (MW)",
    title_x=0.3,
    title_font=dict(size=18),
    height=500
)
            set_sensible_limits(fig_anomalies, df_for_analysis)
            st.plotly_chart(fig_anomalies, use_container_width=True)
        else:
            st.success("✅ No significant anomalies detected based on the current criteria.")
    else:
        st.info("No valid data to perform anomaly detection.")
    
    if compare_mode and uploaded_file_2:
        df_clean_2, sheet_name_2 = process_turbine_data(uploaded_file_2, conversion_needed, sample_fraction, start_date, end_date)
        
        if df_clean_2 is not None:
            sheet_suffix_2 = f" - {sheet_name_2}" if sheet_name_2 else ""
            
            st.header("📊 Turbine Comparison")
            
            fig_compare = go.Figure()
            
            fig_compare.add_trace(go.Scatter(
                x=df_clean_1['Wind_Raw'], y=df_clean_1['Power_Raw'],
                mode='markers',
                name=f'Turbine 1 ({uploaded_file.name}{sheet_suffix_1})',
                marker=dict(color='blue', opacity=transparency, size=5)
            ))
            
            fig_compare.add_trace(go.Scatter(
                x=df_clean_2['Wind_Raw'], y=df_clean_2['Power_Raw'],
                mode='markers',
                name=f'Turbine 2 ({uploaded_file_2.name}{sheet_suffix_2})',
                marker=dict(color='green', opacity=transparency, size=5)
            ))
            
            power_reference = get_expected_power(np.linspace(0, cut_out, 100))
            if density_correction:
                power_reference = apply_density_correction(power_reference, air_density)
            fig_compare.add_trace(go.Scatter(
                x=np.linspace(0, cut_out, 100), y=power_reference,
                mode='lines',
                name='Official V90 2MW Curve',
                line=dict(color='red', width=3)
            ))
            
            fig_compare.update_layout(
    title=f"Power Curve Comparison: T1{sheet_suffix_1} vs T2{sheet_suffix_2}",
    xaxis_title="Wind Speed (m/s)",
    yaxis_title="Power (MW)",
    title_x=0.3,
    title_font=dict(size=18),
    height=600
)
            set_sensible_limits(fig_compare, pd.concat([df_clean_1, df_clean_2]))
            st.plotly_chart(fig_compare, use_container_width=True)
        else:
            st.warning("Could not process file for Turbine 2. Please check its format.")

    st.subheader("📊 Data Quality")
    quality_counts = df_clean_1['Data_Quality'].value_counts()
    colors = {'Valid Data': 'green', 'Corrected to 0': 'red', 'Invalid Data': 'gray'}
    fig1 = px.pie(values=quality_counts.values, names=quality_counts.index, color=quality_counts.index,
                  color_discrete_map=colors,
                  title=f"Data Quality – V90 2MW MK7 Turbine{sheet_suffix_1}")
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    fig1.update_layout(height=500, title_x=0.3, title_font=dict(size=18), margin=dict(t=50, b=0, l=0, r=0))
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📈 Power Time Series")
    fig2 = px.scatter(df_clean_1, x='Timestamp', y='Power_Raw', color='Data_Quality',
                      color_discrete_map={'Valid Data': 'green', 'Corrected to 0': 'red', 'Invalid Data': 'gray'},
                      opacity=0.5,
                      labels={'Timestamp': 'Time', 'Power_Raw': 'Power (MW)', 'Data_Quality': 'Data Quality'},
                      title=f"Time Series – V90 2MW MK7 Turbine{sheet_suffix_1}")
    fig2.update_layout(height=500, title_x=0.3, title_font=dict(size=18))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("⚡ Power Curve")
    fig3 = go.Figure()
    
    for quality, group in df_clean_1.groupby('Data_Quality'):
        fig3.add_trace(go.Scatter(
            x=group['Wind_Raw'], y=group['Power_Raw'],
            mode='markers',
            name=quality if quality == 'Valid Data' else None,
            marker=dict(color=colors.get(quality, 'blue'), opacity=transparency, size=5),
            hovertemplate="Wind Speed: %{x:.2f} m/s<br>Power: %{y:.2f} MW<br>Data Quality: " + quality + "<extra></extra>"
        ))
    
    power_reference = get_expected_power(np.linspace(0, cut_out, 100))
    if density_correction:
        power_reference = apply_density_correction(power_reference, air_density)
    fig3.add_trace(go.Scatter(
        x=np.linspace(0, cut_out, 100), y=power_reference,
        mode='lines',
        name='Official V90 2MW Curve',
        line=dict(color='red', width=3)
    ))
    
    fig3.add_trace(go.Scatter(
        x=power_curve_df['WS_m_s'], y=power_curve_df['Production_MW'],
        mode='markers',
        name='Official V90 2MW Points',
        marker=dict(color='blue', size=8, symbol='circle')
    ))
    
    fig3.add_vline(x=cut_in, line_dash="dash", line_color="gray", 
                  annotation_text=f'Cut-in ({cut_in} m/s)', 
                  annotation_position="right",
                  annotation_y=0.5)
    fig3.add_vline(x=rated_start, line_dash="dash", line_color="green", 
                  annotation_text=f'Rated ({rated_start} m/s)', 
                  annotation_position="right",
                  annotation_y=0.5)
    fig3.add_vline(x=cut_out, line_dash="dash", line_color="red", 
                  annotation_text=f'Cut-out ({cut_out} m/s)', 
                  annotation_position="right",
                  annotation_y=0.5)
    fig3.add_hline(y=rated_power, line_dash="dash", line_color="purple", 
                  annotation_text=f'Rated Power ({rated_power} MW)', 
                  annotation_position="right",
                  annotation_y=0.5)

    fig3.update_layout(
    title=f"Power Curve – V90 2MW MK7 Turbine{sheet_suffix_1} (Air density: {air_density} kg/m³)",
    xaxis_title="Wind Speed (m/s)",
    yaxis_title="Power (MW)",
    title_x=0.3,
    title_font=dict(size=18),
    height=600
)
    fig3.update_xaxes(range=[0, 30])
    set_sensible_limits(fig3, df_clean_1)
    st.plotly_chart(fig3, use_container_width=True)

    df_valid = df_for_analysis_1[df_for_analysis_1['Data_Quality'] == 'Valid Data']
    df_corrected = df_for_analysis_1[df_for_analysis_1['Data_Quality'] == 'Corrected to 0']

    if not df_valid.empty:
        total_valid_points = len(df_valid)
        operating_points = len(df_valid[df_valid['Power_Raw'] > 0])
        availability = (operating_points / total_valid_points) * 100 if total_valid_points > 0 else 0

        st.subheader("⚡ Power Curve (Valid Data)")
        fig4 = go.Figure()

        fig4.add_trace(go.Scatter(
            x=df_valid['Wind_Raw'], y=df_valid['Power_Raw'],
            mode='markers',
            name='Valid Data',
            marker=dict(color='green', opacity=transparency, size=5),
            hovertemplate="Wind Speed: %{x:.2f} m/s<br>Power: %{y:.2f} MW<extra></extra>"
        ))

        fig4.add_trace(go.Scatter(
            x=df_corrected['Wind_Raw'], y=df_corrected['Power_Raw'],
            mode='markers',
            name='Corrected to 0',
            marker=dict(color='red', opacity=transparency, size=5),
            hovertemplate="Wind Speed: %{x:.2f} m/s<br>Power: %{y:.2f} MW<extra></extra>"
        ))
        
        bins = np.arange(0, cut_out + bin_size, bin_size)
        df_for_analysis['Wind_bin'] = pd.cut(df_for_analysis['Wind_Raw'], bins)
        binned = (
            df_for_analysis.groupby('Wind_bin')
                .agg(Wind_mean=('Wind_Raw', 'mean'),
                     Power_mean=('Power_Raw', 'mean'),
                     Power_std=('Power_Raw', 'std'),
                     Count=('Power_Raw', 'count'))
                .reset_index()
                .query('Count > 0')
        )
        
        fig4.add_trace(go.Scatter(
            x=np.linspace(0, cut_out, 100), y=get_expected_power(np.linspace(0, cut_out, 100)),
            mode='lines',
            name='Official V90 2MW Curve',
            line=dict(color='red', width=3)
        ))
        fig4.add_trace(go.Scatter(
            x=power_curve_df['WS_m_s'], y=power_curve_df['Production_MW'],
            mode='markers',
            name='Official V90 2MW Points',
            marker=dict(color='blue', size=8, symbol='circle')
        ))
        
        if not binned.empty:
            fig4.add_trace(go.Scatter(
                x=binned['Wind_mean'], y=binned['Power_mean'],
                mode='lines+markers',
                name='Binned Averages',
                line=dict(color='black', width=2),
                marker=dict(size=6, symbol='square')
            ))

        fig4.add_vline(x=cut_in, line_dash="dash", line_color="gray", 
                      annotation_text=f'Cut-in ({cut_in} m/s)', 
                      annotation_position="right",
                      annotation_y=0.5)
        fig4.add_vline(x=rated_start, line_dash="dash", line_color="green", 
                      annotation_text=f'Rated ({rated_start} m/s)', 
                      annotation_position="right",
                      annotation_y=0.5)
        fig4.add_vline(x=cut_out, line_dash="dash", line_color="red", 
                      annotation_text=f'Cut-out ({cut_out} m/s)', 
                      annotation_position="right",
                      annotation_y=0.5)
        fig4.add_hline(y=rated_power, line_dash="dash", line_color="purple", 
                      annotation_text=f'Rated Power ({rated_power} MW)', 
                      annotation_position="right",
                      annotation_y=0.5)

        fig4.update_layout(
    title=f"Power Curve (Valid Data) – V90 2MW MK7 Turbine{sheet_suffix_1}",
    xaxis_title="Wind Speed (m/s)",
    yaxis_title="Power (MW)",
    title_x=0.3,
    title_font=dict(size=18),
    height=600
)
        fig4.update_xaxes(range=[0, 30])
        set_sensible_limits(fig4, df_valid)
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("📅 Daily Efficiency (Valid Data)")
        
        valid_data = df_valid.copy()
        valid_data['Expected_Power'] = get_expected_power(valid_data['Wind_Raw'])
        if density_correction:
            valid_data['Expected_Power'] = apply_density_correction(valid_data['Expected_Power'], air_density)
        valid_data['Efficiency'] = np.where(
            valid_data['Expected_Power'] > 0,
            (valid_data['Power_Raw'] / valid_data['Expected_Power']) * 100,
            0
        )
        daily_eff = valid_data.set_index('Timestamp')['Efficiency'].resample('D').mean().reset_index()
        daily_eff['Efficiency_Status'] = np.where(daily_eff['Efficiency'] >= 100, 'Above Expected', 'Below Expected')

        fig5 = go.Figure()
        
        fig5.add_trace(go.Scatter(
            x=daily_eff['Timestamp'], y=daily_eff['Efficiency'],
            mode='lines+markers',
            name='Daily Average',
            line=dict(color='blue')
        ))
        
        fig5.add_trace(go.Scatter(
            x=daily_eff['Timestamp'], y=[100] * len(daily_eff),
            mode='lines',
            name='100% Efficiency',
            line=dict(color='red', dash='dash')
        ))
        
        fig5.update_layout(
    title=f"Daily Efficiency – V90 2MW MK7 Turbine{sheet_suffix_1}",
    xaxis_title='Date',
    yaxis_title='Efficiency (%)',
    title_x=0.3,
    title_font=dict(size=18),
    height=500
)
        st.plotly_chart(fig5, use_container_width=True)

    else:
        st.info("No valid data points found to calculate daily efficiency.")

    st.subheader("💾 Data Export")
    export_df = df_clean_1.copy()
    if not df_valid.empty:
        if 'Expected_Power' in df_valid.columns and 'Efficiency' in df_valid.columns:
            export_df = pd.merge(export_df, df_valid[['Timestamp', 'Expected_Power', 'Efficiency']],
                                 on='Timestamp', how='left')
    if 'is_anomaly' in df_for_analysis.columns:
        export_df = pd.merge(export_df, df_for_analysis[['Timestamp', 'is_anomaly']],
                             on='Timestamp', how='left')

    export_df['Timestamp'] = pd.to_datetime(export_df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, sheet_name='Processed_Data', index=False)
        if 'binned' in locals():
            binned.to_excel(writer, sheet_name='Bin_Averages', index=False)
    excel_buffer.seek(0)

    st.download_button(
        label="💾 Download Processed Data (Excel)",
        data=excel_buffer.getvalue(),
        file_name=f"SCADA_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    total_points = len(df_clean_1)
    valid_points = len(df_valid)
    valid_percentage = (valid_points / total_points * 100) if total_points > 0 else 0
    
    avg_wind = df_clean_1['Wind_Raw'].mean() if not df_clean_1['Wind_Raw'].empty else 'N/A'
    std_wind = df_clean_1['Wind_Raw'].std() if not df_clean_1['Wind_Raw'].empty else 'N/A'
    
    avg_power = df_valid['Power_Raw'].mean() if valid_points > 0 else 'N/A'
    std_power = df_valid['Power_Raw'].std() if valid_points > 0 else 'N/A'

    st.subheader("📊 Final Statistics")
    cols = st.columns(5)
    cols[0].metric("Total Data Points", total_points)
    cols[1].metric("Valid Data Points", valid_points, f"{valid_percentage:.1f}%")
    cols[2].metric("Average Wind Speed", f"{avg_wind:.2f} m/s" if isinstance(avg_wind, float) else "N/A",
                   f"±{std_wind:.2f}" if isinstance(std_wind, float) else "")
    cols[3].metric("Average Power", f"{avg_power:.3f} MW" if isinstance(avg_power, float) else "N/A",
                   f"±{std_power:.3f}" if isinstance(std_power, float) else "")
    if 'availability' in locals():
        cols[4].metric("Availability", f"{availability:.1f}%")
    else:
        cols[4].metric("Availability", "N/A")

    st.markdown("---")
    st.subheader("📤 Export Full Analysis")
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Power Curve Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2e86c1; }}
        h2 {{ color: #1a5276; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .plot {{ width: 100%; height: auto; margin-bottom: 30px; }}
        .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌬️ Power Curve Analysis - Vestas V90 2MW MK7</h1>
        <p>Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>File analyzed: {uploaded_file.name}{' - Sheet: ' + sheet_name_1 if sheet_name_1 else ''}</p>
        
        <h2>Data Quality Overview</h2>
        <div class="plot">{fig1.to_html(full_html=False)}</div>
        
        <h2>Power Time Series</h2>
        <div class="plot">{fig2.to_html(full_html=False)}</div>
        
        <h2>Power Curve Analysis</h2>
        <div class="plot">{fig3.to_html(full_html=False)}</div>
        
        <h2>Power Curve (Valid Data)</h2>
        <div class="plot">{fig4.to_html(full_html=False) if 'fig4' in locals() else '<p>No valid data available</p>'}</div>
        
        <h2>Daily Efficiency</h2>
        <div class="plot">{fig5.to_html(full_html=False) if 'fig5' in locals() else '<p>No valid data available</p>'}</div>
        
        <h2>Turbulence Analysis</h2>
        <div class="plot">{fig_turbulence.to_html(full_html=False) if 'fig_turbulence' in locals() else '<p>No valid data available for turbulence analysis</p>'}</div>
        
        <h2>Anomaly Detection</h2>
        <div class="plot">{fig_anomalies.to_html(full_html=False) if 'fig_anomalies' in locals() else '<p>No anomalies detected or no valid data available</p>'}</div>

        <h2>Final Statistics</h2>
        <ul>
            <li>Total Data Points: {total_points}</li>
            <li>Valid Data Points: {valid_points} ({valid_percentage:.1f}%)</li>
            <li>Average Wind Speed: {avg_wind:.2f} ± {std_wind:.2f} m/s</li>
            <li>Average Power: {avg_power:.3f} ± {std_power:.3f} MW</li>
            <li>Availability: {availability:.1f}%</li>
        </ul>
        
        <div class="footer">
            <p>Report generated using Advanced SCADA Analysis Tool by ERS</p>
        </div>
    </div>
</body>
</html>
"""

    st.download_button(
        label="📄 Download Full Report (HTML)",
        data=html_content,
        file_name=f"SCADA_Analysis_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html"
    )
    
else:
    st.info("🔽 Please upload a SCADA data file to begin analysis")
