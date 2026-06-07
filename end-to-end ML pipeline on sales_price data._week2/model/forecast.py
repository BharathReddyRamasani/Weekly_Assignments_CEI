import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import clean_data

def generate_forecast():
    # Load data from local
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'tesla_data.csv')
    df = pd.read_csv(data_path)
    df_clean = clean_data(df)
    
    df_ts = df_clean.copy()
    df_ts["Date"] = pd.to_datetime(df_ts["Year"].astype(str) + "-" + df_ts["Month"].astype(str) + "-01")
    monthly_series = df_ts.groupby("Date")["Estimated_Deliveries"].sum()
    
    # Train test split for the time series
    train = monthly_series[:-12]
    test = monthly_series[-12:]
    
    print("Fitting Holt-Winters Exponential Smoothing...")
    hw_model = ExponentialSmoothing(
        train,
        trend='add',
        seasonal='add',
        seasonal_periods=12
    )
    hw_result = hw_model.fit(optimized=True)
    
    forecast = hw_result.forecast(steps=12)
    forecast.index = test.index
    
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_series, label="Actual")
    plt.plot(forecast.index, forecast.values, label="Forecast")
    plt.title("Tesla Deliveries Forecast (Holt-Winters)")
    plt.legend()
    
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    plt.savefig(os.path.join(outputs_dir, 'forecast_plot.png'))
    print("Forecast plot saved.")

if __name__ == "__main__":
    generate_forecast()
