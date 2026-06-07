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

DATA_PATH      = os.path.join(BASE_DIR, 'data', 'tesla_data.csv')
MODEL_PATH     = os.path.join(BASE_DIR, 'model', 'tesla_model.pkl')
SCALER_PATH    = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
METRICS_PATH   = os.path.join(BASE_DIR, 'outputs', 'model_metrics.csv')
FORECAST_PATH  = os.path.join(BASE_DIR, 'outputs', 'forecast_plot.png')
IMPORTANCE_PATH= os.path.join(BASE_DIR, 'outputs', 'feature_importances.csv')
PARAMS_PATH    = os.path.join(BASE_DIR, 'outputs', 'best_params.json')

st.set_page_config(page_title="Tesla Intelligence", page_icon="⚡", layout="wide")

# ──────────────────────────── STYLING ────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"] { background:#1a1a2e; border-radius:10px; padding:12px; }
[data-testid="stMetricLabel"] { font-size:0.75rem; color:#aaa; }
[data-testid="stMetricValue"] { font-size:1.5rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

H = 480  # standard chart height

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

st.title("⚡ Tesla Delivery Predictive Analytics")

if df.empty:
    st.error("Data source unavailable.")
    st.stop()

# ─────────────────────── KPI HEADER ───────────────────────
total_deliveries = df['Estimated_Deliveries'].sum()
latest_delivery  = df.iloc[-1]['Estimated_Deliveries']
prev_delivery    = df.iloc[-13]['Estimated_Deliveries'] if len(df) > 13 else df.iloc[-2]['Estimated_Deliveries']
yoy_growth       = ((latest_delivery - prev_delivery) / prev_delivery) * 100
avg_price        = df['Avg_Price_USD'].mean() if 'Avg_Price_USD' in df.columns else 0
co2_saved        = df['CO2_Saved_tons'].sum() if 'CO2_Saved_tons' in df.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Deliveries",    f"{total_deliveries:,.0f}")
c2.metric("Latest Month",        f"{latest_delivery:,.0f}",  f"{yoy_growth:.1f}% YoY")
c3.metric("Data Points",         f"{len(df):,}")
c4.metric("Avg Price (USD)",     f"${avg_price:,.0f}")
c5.metric("CO₂ Saved (tons)",    f"{co2_saved:,.0f}")

# ─────────────────────── SIDEBAR ──────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/bb/Tesla_T_symbol.svg", width=60)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 EDA",
    "🧹 Data Engineering",
    "⚙️ ML Evaluation",
    "🔮 Forecast",
    "🕹️ Prediction"
], key="main_nav")

# ══════════════════════════════════════════════
#  📊  EDA
# ══════════════════════════════════════════════
if page == "📊 EDA":
    st.header("📊 Exploratory Data Analysis")

    # 1. Time-series
    st.subheader("Delivery Trend Over Time")
    monthly = df.groupby('Date')['Estimated_Deliveries'].sum().reset_index()
    fig = px.area(monthly, x='Date', y='Estimated_Deliveries',
                  color_discrete_sequence=["#E82127"])
    fig.update_layout(height=H, xaxis_title="", yaxis_title="Deliveries",
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font_color='white')
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    # 2. Deliveries by Model
    with col1:
        st.subheader("Deliveries by Model")
        model_df = df.groupby('Model')['Estimated_Deliveries'].sum().reset_index()
        fig2 = px.bar(model_df, x='Model', y='Estimated_Deliveries',
                      color='Model', color_discrete_sequence=px.colors.qualitative.Bold)
        fig2.update_layout(height=H, showlegend=False,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           font_color='white')
        st.plotly_chart(fig2, use_container_width=True)

    # 3. Deliveries by Region
    with col2:
        st.subheader("Deliveries by Region")
        region_df = df.groupby('Region')['Estimated_Deliveries'].sum().reset_index()
        fig3 = px.pie(region_df, names='Region', values='Estimated_Deliveries',
                      hole=0.4, color_discrete_sequence=px.colors.qualitative.Bold)
        fig3.update_layout(height=H, paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig3, use_container_width=True)

    # 4. Feature Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    cols = corr.columns.tolist()
    fig_heat = px.imshow(
        corr.values,
        x=cols, y=cols,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1
    )
    fig_heat.update_layout(
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis=dict(tickangle=-45)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # 5. Avg Price by Year
    st.subheader("Average Price Trend by Year")
    price_df = df.groupby('Year')['Avg_Price_USD'].mean().reset_index()
    fig5 = px.line(price_df, x='Year', y='Avg_Price_USD', markers=True,
                   color_discrete_sequence=["#FFD700"])
    fig5.update_layout(height=H, plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════
#  🧹  DATA ENGINEERING
# ══════════════════════════════════════════════
elif page == "🧹 Data Engineering":
    st.header("🧹 Data Engineering")

    df_clean  = clean_data(df)
    df_capped = cap_outliers(df_clean)
    df_feats  = create_features(df_capped)

    # Outlier boxplots
    st.subheader("Outlier Detection — Before vs After IQR Capping")
    num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    sel_col = st.selectbox("Select Feature", num_cols)

    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=df_clean[sel_col],  name="Before Capping",
                             marker_color="#E82127", boxmean=True))
    fig_box.add_trace(go.Box(y=df_capped[sel_col], name="After Capping",
                             marker_color="#00C851", boxmean=True))
    fig_box.update_layout(height=H, title=f"{sel_col} — Outlier Capping",
                          plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_box, use_container_width=True)

    col1, col2 = st.columns(2)

    # Distribution histogram
    with col1:
        st.subheader("Distribution After Capping")
        fig_hist = px.histogram(df_capped, x=sel_col, nbins=40,
                                color_discrete_sequence=["#E82127"])
        fig_hist.update_layout(height=H, plot_bgcolor='rgba(0,0,0,0)',
                               paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_hist, use_container_width=True)

    # Lag feature scatter
    with col2:
        st.subheader("Lag-1 vs Deliveries")
        if 'Lag_1' in df_feats.columns:
            fig_lag = px.scatter(df_feats, x='Lag_1', y='Estimated_Deliveries',
                                 opacity=0.6, color_discrete_sequence=["#00C851"],
                                 trendline="ols")
            fig_lag.update_layout(height=H, plot_bgcolor='rgba(0,0,0,0)',
                                  paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_lag, use_container_width=True)

    # Rolling means
    st.subheader("Feature Engineering — Rolling Averages vs Actual")
    if 'Rolling_Mean_3' in df_feats.columns and 'Date' in df_feats.columns:
        fig_roll = px.line(df_feats, x='Date',
                           y=['Estimated_Deliveries', 'Rolling_Mean_3', 'Rolling_Mean_6'],
                           color_discrete_map={
                               'Estimated_Deliveries': '#E82127',
                               'Rolling_Mean_3': '#FFD700',
                               'Rolling_Mean_6': '#00C851'
                           })
        fig_roll.update_layout(height=H, plot_bgcolor='rgba(0,0,0,0)',
                               paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                               legend_title="Series")
        st.plotly_chart(fig_roll, use_container_width=True)

    # Preprocessing summary table
    st.subheader("Preprocessed Data Sample")
    st.dataframe(df_capped.head(20), use_container_width=True, height=350)

# ══════════════════════════════════════════════
#  ⚙️  ML EVALUATION
# ══════════════════════════════════════════════
elif page == "⚙️ ML Evaluation":
    st.header("⚙️ ML Model Evaluation")

    if os.path.exists(METRICS_PATH):
        metrics_df = pd.read_csv(METRICS_PATH)

        # Model comparison bar
        st.subheader("Model Comparison — MAE")
        fig_bar = px.bar(metrics_df.sort_values("MAE"), x="MAE", y="Model",
                         orientation='h', color="MAE",
                         color_continuous_scale="Reds", text_auto=".0f")
        fig_bar.update_layout(height=H, plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_bar, use_container_width=True)

        col1, col2 = st.columns(2)

        # RMSE comparison
        with col1:
            st.subheader("Model Comparison — RMSE")
            fig_rmse = px.bar(metrics_df.sort_values("RMSE"), x="RMSE", y="Model",
                              orientation='h', color="RMSE",
                              color_continuous_scale="Oranges", text_auto=".0f")
            fig_rmse.update_layout(height=H, plot_bgcolor='rgba(0,0,0,0)',
                                   paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_rmse, use_container_width=True)

        # R² comparison
        with col2:
            st.subheader("Model Comparison — R² Score")
            fig_r2 = px.bar(metrics_df.sort_values("R2", ascending=False),
                            x="R2", y="Model", orientation='h', color="R2",
                            color_continuous_scale="Greens", text_auto=".4f")
            fig_r2.update_layout(height=H, plot_bgcolor='rgba(0,0,0,0)',
                                 paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_r2, use_container_width=True)

        # Metrics table
        st.subheader("Full Metrics Table")
        st.dataframe(
            metrics_df.style
              .highlight_min(subset=['MAE', 'RMSE'], color='#1a6b1a')
              .highlight_max(subset=['R2'],            color='#1a6b1a')
              .format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R2": "{:.4f}"}),
            use_container_width=True, height=250
        )
    else:
        st.warning("Model metrics file not found.")

    col3, col4 = st.columns(2)

    # Hyperparameters
    with col3:
        st.subheader("Champion Model Hyperparameters")
        if os.path.exists(PARAMS_PATH):
            with open(PARAMS_PATH, 'r') as f:
                params = json.load(f)
            st.json(params)
        else:
            st.info("No hyperparameter file found.")

    # Feature Importance
    with col4:
        st.subheader("Feature Importances")
        if os.path.exists(IMPORTANCE_PATH):
            fi_df = pd.read_csv(IMPORTANCE_PATH).sort_values("Importance", ascending=True)
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation='h',
                            color="Importance", color_continuous_scale="Reds",
                            text_auto=".4f")
            fig_fi.update_layout(height=H, plot_bgcolor='rgba(0,0,0,0)',
                                 paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("No feature importances file found.")

# ══════════════════════════════════════════════
#  🔮  FORECAST
# ══════════════════════════════════════════════
elif page == "🔮 Forecast":
    st.header("🔮 Holt-Winters Forecast")

    # Static forecast image
    if os.path.exists(FORECAST_PATH):
        st.subheader("12-Month Forecast Plot")
        st.image(FORECAST_PATH, use_container_width=True)
    else:
        st.info("Forecast image not found — generating interactive forecast below.")

    # Interactive Holt-Winters forecast
    st.subheader("Interactive 24-Month Forecast")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    monthly = df.groupby('Date')['Estimated_Deliveries'].sum()
    monthly.index = pd.DatetimeIndex(monthly.index).to_period('M').to_timestamp()

    try:
        hw = ExponentialSmoothing(monthly.values, trend='add', seasonal='add',
                                  seasonal_periods=12).fit(optimized=True)
        fc = hw.forecast(24)
        fc_dates = pd.date_range(start=monthly.index[-1], periods=25, freq='MS')[1:]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=monthly.index, y=monthly.values,
                                    name="Historical", line=dict(color="#E82127", width=2)))
        fig_fc.add_trace(go.Scatter(x=fc_dates, y=fc,
                                    name="Forecast", line=dict(color="#FFD700", width=2, dash='dot'),
                                    fill='tozeroy', fillcolor='rgba(255,215,0,0.1)'))
        fig_fc.update_layout(height=H+100, plot_bgcolor='rgba(0,0,0,0)',
                             paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                             xaxis_title="Date", yaxis_title="Deliveries",
                             legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_fc, use_container_width=True)

        # Forecast table
        st.subheader("Forecasted Values")
        fc_df = pd.DataFrame({'Date': fc_dates, 'Predicted_Deliveries': fc.astype(int)})
        st.dataframe(fc_df, use_container_width=True, height=300)
    except Exception as e:
        st.error(f"Forecast error: {e}")

# ══════════════════════════════════════════════
#  🕹️  PREDICTION
# ══════════════════════════════════════════════
elif page == "🕹️ Prediction":
    st.header("🕹️ Live Delivery Prediction")

    col1, col2 = st.columns(2)
    with col1:
        input_year  = st.number_input("Year",  min_value=2015, max_value=2035, value=2028)
    with col2:
        input_month = st.selectbox("Month", options=list(range(1, 13)),
                                   format_func=lambda m: pd.Timestamp(2000, m, 1).strftime('%B'))

    if st.button("🔍 Predict Deliveries", type="primary"):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        try:
            last_date   = df['Date'].max()
            target_date = pd.to_datetime(f"{input_year}-{input_month:02d}-01")

            if target_date <= last_date:
                mask = df['Date'] == target_date
                if mask.any():
                    val = df.loc[mask, 'Estimated_Deliveries'].sum()
                    st.success(f"✅ Actual Deliveries on record: **{val:,.0f} units**")
                else:
                    st.warning("No exact record found for that date.")
            else:
                months_ahead  = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
                monthly       = df.groupby('Date')['Estimated_Deliveries'].sum()
                monthly.index = pd.DatetimeIndex(monthly.index).to_period('M').to_timestamp()
                hw = ExponentialSmoothing(monthly.values, trend='add', seasonal='add',
                                         seasonal_periods=12).fit(optimized=True)
                fc = hw.forecast(steps=months_ahead)
                prediction = max(0, int(fc[-1]))

                st.success(f"🔮 Predicted Deliveries for **{pd.Timestamp(input_year, input_month, 1).strftime('%B %Y')}**: **{prediction:,.0f} units**")

                # Show forecast chart up to selected date
                fc_full  = hw.forecast(steps=months_ahead)
                fc_dates = pd.date_range(start=monthly.index[-1], periods=months_ahead + 1, freq='MS')[1:]
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=monthly.index[-24:], y=monthly.values[-24:],
                                              name="Recent History", line=dict(color="#E82127", width=2)))
                fig_pred.add_trace(go.Scatter(x=fc_dates, y=fc_full,
                                              name="Forecast", line=dict(color="#FFD700", width=2, dash='dot')))
                fig_pred.add_vline(x=target_date.strftime("%Y-%m-%d"), line_dash="dash", line_color="white",
                                   annotation_text=f"Target: {prediction:,}", annotation_position="top right")
                fig_pred.update_layout(height=H, plot_bgcolor='rgba(0,0,0,0)',
                                       paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_pred, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
