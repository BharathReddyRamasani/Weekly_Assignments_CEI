import json

notebook_path = 'd:/celebal/w2.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if '# Forecast future deliveries' in source:
            new_source = """# Forecast future deliveries using Holt-Winters Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

train = monthly_series[:-12]
test = monthly_series[-12:]

# Exponential Smoothing is excellent for data with strong trend and seasonality
logging.info("Using Holt-Winters Exponential Smoothing")
hw_model = ExponentialSmoothing(
    train,
    trend='add',
    seasonal='add',
    seasonal_periods=12
)
hw_result = hw_model.fit(optimized=True)

forecast = hw_result.forecast(steps=12)
forecast.index = test.index

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print("Holt-Winters MAE:", mean_absolute_error(test, forecast))
print("Holt-Winters RMSE:", np.sqrt(mean_squared_error(test, forecast)))
print("Holt-Winters MAPE:", np.mean(np.abs((test - forecast) / test)) * 100, "%")

forecast"""
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            cell['source'][-1] = cell['source'][-1].rstrip('\n')

        # Also update the plot title to reflect Holt-Winters
        elif '# Plot actual vs forecast' in source:
            new_source = source.replace("Tesla Deliveries Forecast", "Tesla Deliveries Forecast (Holt-Winters)")
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            cell['source'][-1] = cell['source'][-1].rstrip('\n')


with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("ARIMA replaced with Holt-Winters Exponential Smoothing")
