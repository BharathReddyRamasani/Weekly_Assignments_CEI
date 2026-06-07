import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from utils.preprocessing import clean_data, cap_outliers
from utils.feature_engineering import create_features

DATA_PATH = os.path.join(BASE_DIR, 'data', 'tesla_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'tesla_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'outputs', 'model_metrics.csv')
FORECAST_PLOT_PATH = os.path.join(BASE_DIR, 'outputs', 'forecast_plot.png')
IMPORTANCE_PATH = os.path.join(BASE_DIR, 'outputs', 'feature_importances.csv')
PARAMS_PATH = os.path.join(BASE_DIR, 'outputs', 'best_params.json')

st.set_page_config(page_title="Tesla Intelligence", page_icon="⚡", layout="wide")

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if 'Year' in df.columns and 'Month' in df.columns:
            df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
            df = df.sort_values('Date').reset_index(drop=True)
        return df
    return pd.DataFrame()

df = load_data()

st.title("Tesla Delivery Predictive Analytics")

if not df.empty:
    col1, col2, col3, col4 = st.columns(4)
    total_deliveries = df['Estimated_Deliveries'].sum()
    latest_delivery = df.iloc[-1]['Estimated_Deliveries']
    prev_year_delivery = df.iloc[-13]['Estimated_Deliveries'] if len(df) > 13 else df.iloc[-2]['Estimated_Deliveries']
    yoy_growth = ((latest_delivery - prev_year_delivery) / prev_year_delivery) * 100

    col1.metric("Total Deliveries", f"{total_deliveries:,.0f}")
    col2.metric("Latest Month", f"{latest_delivery:,.0f}", f"{yoy_growth:.1f}% YoY")
    col3.metric("Data Points", f"{len(df)}")
    
    if os.path.exists(METRICS_PATH):
        metrics_df = pd.read_csv(METRICS_PATH)
        best_mae = metrics_df['MAE'].min()
        col4.metric(f"Best Model MAE", f"±{best_mae:,.0f}")

    # --- SIDEBAR NAVIGATION ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio("Go to", [
        "📊 EDA", 
        "🧹 Data Engineering", 
        "⚙️ ML Evaluation", 
        "🔮 Forecast",
        "🕹️ Prediction"
    ])

    if page == "📊 EDA":
        st.subheader("Time-Series Trajectory")
        fig = px.line(df, x='Date', y='Estimated_Deliveries', markers=True, line_shape='spline')
        fig.update_layout(xaxis_title="", yaxis_title="Deliveries")
        fig.update_traces(line_color="#E82127")
        st.plotly_chart(fig, use_container_width=True)
        
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.subheader("Feature Correlation Heatmap")
            numeric_df = df.select_dtypes(include=np.number)
            corr = numeric_df.corr()
            fig_heat = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_heat, use_container_width=True)
            
        with col_eda2:
            st.subheader("Scatter Matrix (Key Features)")
            scatter_cols = ['Estimated_Deliveries', 'Year', 'Month']
            scatter_cols = [c for c in scatter_cols if c in df.columns]
            fig_scatter = px.scatter_matrix(df, dimensions=scatter_cols, color='Estimated_Deliveries')
            fig_scatter.update_traces(diagonal_visible=False)
            st.plotly_chart(fig_scatter, use_container_width=True)

    elif page == "🧹 Data Engineering":
        st.subheader("Outlier Mitigation (IQR Capping)")
        df_clean = clean_data(df)
        df_capped = cap_outliers(df_clean)
        
        num_cols = df_clean.select_dtypes(include=np.number).columns
        sel_col = st.selectbox("Select Feature for Outlier Analysis", num_cols)
        
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=df_clean[sel_col], name="Before Capping", marker_color="red"))
        fig_box.add_trace(go.Box(y=df_capped[sel_col], name="After Capping", marker_color="green"))
        fig_box.update_layout(title="Outlier Detection Boxplots")
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.subheader("Feature Engineering Visualization")
        df_features = create_features(df_capped)
        if 'Lag_1' in df_features.columns:
            fig_feat = px.line(df_features, x='Date', y=['Estimated_Deliveries', 'Rolling_Mean_3'], 
                               title="Actual vs Rolling Mean (3 Months)")
            st.plotly_chart(fig_feat, use_container_width=True)

    elif page == "⚙️ ML Evaluation":
        st.subheader("Algorithm Performance")
        if os.path.exists(METRICS_PATH):
            metrics_df = pd.read_csv(METRICS_PATH)
            fig_bar = px.bar(metrics_df.sort_values("MAE"), x="MAE", y="Model", orientation='h', color="MAE", color_continuous_scale="Reds")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(metrics_df.style.highlight_min(subset=['MAE', 'RMSE'], color='#90EE90').format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R2": "{:.4f}"}), use_container_width=True)
        
        col_ml1, col_ml2 = st.columns(2)
        with col_ml1:
            st.subheader("Champion Model Hyperparameters")
            if os.path.exists(PARAMS_PATH):
                with open(PARAMS_PATH, 'r') as f:
                    params = json.load(f)
                st.json(params)
            else:
                st.info("No hyperparameters found.")
                
        with col_ml2:
            st.subheader("Feature Importances")
            if os.path.exists(IMPORTANCE_PATH):
                fi_df = pd.read_csv(IMPORTANCE_PATH).sort_values("Importance", ascending=True)
                fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation='h', title="Random Forest Feature Importance")
                st.plotly_chart(fig_fi, use_container_width=True)
                st.dataframe(fi_df.sort_values("Importance", ascending=False), use_container_width=True)
            else:
                st.info("No feature importances found.")

    elif page == "🔮 Forecast":
        st.subheader("Holt-Winters 12-Month Forecast")
        if os.path.exists(FORECAST_PLOT_PATH):
            st.image(Image.open(FORECAST_PLOT_PATH), use_container_width=True)

    elif page == "🕹️ Prediction":
        st.subheader("Live Delivery Prediction")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            input_year = st.number_input("Year", min_value=2015, max_value=2030, value=2028)
        with col_p2:
            input_month = st.selectbox("Month", options=list(range(1, 13)))
        
        if st.button("Predict Deliveries", type="primary"):
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            try:
                last_date = df['Date'].max()
                target_date = pd.to_datetime(f"{input_year}-{input_month}-01")
                if target_date <= last_date:
                    val = df.loc[df['Date'] == target_date, 'Estimated_Deliveries'].values[0]
                    st.success(f"### Actual Past Deliveries: {val:,.0f} units")
                else:
                    months_ahead = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
                    monthly_series = df.groupby('Date')['Estimated_Deliveries'].sum()
                    hw_model = ExponentialSmoothing(monthly_series, trend='add', seasonal='add', seasonal_periods=12).fit(optimized=True)
                    forecast = hw_model.forecast(steps=months_ahead)
                    prediction = forecast.iloc[-1]
                    st.success(f"### Predicted Deliveries: {prediction:,.0f} units")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
else:
    st.error("Data source unavailable.")
