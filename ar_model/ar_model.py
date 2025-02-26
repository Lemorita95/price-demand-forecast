from auxiliary.helpers import PRICE_FILE, DEMAND_FILE, date_slicer, get_daily_data
from auxiliary.load_data import load_price, load_demand
from auxiliary.plot import plot_1_axis
from auxiliary.styles import *

from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


price_datetime, price_value = load_price(PRICE_FILE)
demand_datetime, demand_value = load_demand(DEMAND_FILE)

price_datetime, price_value = get_daily_data(price_datetime, price_value, 'max')
demand_datetime, demand_value = get_daily_data(demand_datetime, demand_value, 'max')

date_in = date(2021, 1, 1)
date_out = date(2021, 5, 1)

dates = np.array([date_in + timedelta(days=i) 
                            for i in range((date_out - date_in).days + 1)], dtype=object)
price = date_slicer(dates, price_datetime, price_value)
demand = date_slicer(dates, demand_datetime, demand_value)

x = dates
y = price

# Visualize the data
plt.figure(figsize=(12, 6))
plt.plot(y, label='Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('Price Data')
plt.show()

# Create lag features up to 5 days
lag_features = np.column_stack([np.roll(y, i) for i in range(1, 6)])

# Replace first 5 rows with NaN-equivalent values since they have invalid lag data
lag_features[:5] = np.nan

# Remove NaN rows (equivalent to dropna in pandas)
valid_rows = ~np.isnan(lag_features).any(axis=1)
y = y[valid_rows]
x = x[valid_rows]
lag_features = lag_features[valid_rows]

# Split into training and testing sets
train_size = int(0.8 * len(y))
x_train = x[:train_size]
x_test = x[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

from statsmodels.graphics.tsaplots import plot_acf
series = y
plot_acf(series)
plt.show()

y_shifted = np.roll(y, 1)  # Equivalent to y.shift(1)
correlation = np.corrcoef(y[1:], y_shifted[1:])[0, 1]  # Exclude the first element to match the shift
print(correlation)

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Create and train the autoregressive model
lag_order = 7 # Adjust this based on the ACF plot
ar_model = AutoReg(y_train, lags=lag_order)
ar_results = ar_model.fit()

# Make predictions on the test set
y_pred = ar_results.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, dynamic=False)
#print(y_pred)

# Calculate MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(x_test ,y_test, label='Actual Price')
plt.plot(x_test, y_pred, label='Predicted Price', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('Price Prediction with Autoregressive Model')
plt.show()

# Define the number of future time steps you want to predict (1 week)
forecast_steps = 30

# Extend the predictions into the future for one year
future_indices = range(len(y_test), len(y_test) + forecast_steps)
future_predictions = ar_results.predict(start=len(y_train), end=len(y_train) + len(y_test) + forecast_steps - 1, dynamic=False)

# Get the last date from x_test
last_date = x_test[-1]

# Create future date indices
future_dates = future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_steps + 1)]

# **Predict on Train Dataset (In-Sample)**
train_pred = ar_results.predict(start=lag_order, end=len(y_train)-1)

# **Predict on Test Dataset (Out-of-Sample)**
test_pred = ar_results.predict(start=len(y_train), end=len(y)-1)

from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(y_train)
print("ADF p-value:", adf_test[1])

x0, y0 = np.concatenate((np.array([x_train[-1]]), x_test)), np.concatenate((np.array([train_pred[-1]]), test_pred))
plot_1_axis(x, y, x0, y0, **style6)