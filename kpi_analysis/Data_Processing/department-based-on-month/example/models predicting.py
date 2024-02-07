#------------- Forcasting using ETS E_S/enterprise sales (Month AVG) ----------
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

avg_rev = pd.read_csv("average_revenue_by_month_e_s.csv",encoding="ISO-8859-1")
avg_rev['YearMonth'] = pd.to_datetime(avg_rev['YearMonth'])
avg_rev.set_index('YearMonth', inplace=True)

# Extract training and test data
train_data = avg_rev.loc[:'2019-06-01', 'Revenue']
test_data = avg_rev.loc['2019-07-01':'2019-12-01', 'Revenue']

model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=9)  # 9 bagus
fit_model = model.fit()

# Make predictions on the test set
predictions = fit_model.forecast(len(test_data))

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data, label='Training Data')
plt.plot(test_data.index, test_data, label='Test Data')
plt.plot(test_data.index, predictions, label='Predictions', color='red')
plt.title('Monthly Average Enterprise Sales Exponential Smoothing Forecasting with Seasonality')
plt.legend()
plt.show()
# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100

print(f'Mean Absolute Percentage Error for median model (MAPE): {mape:.2f}%')

#------------- Forcasting using ETS E_S/enterprise sales (Month median) -------
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

med_rev = pd.read_csv("median_revenue_by_month_e_s.csv", encoding="ISO-8859-1")
med_rev['YearMonth'] = pd.to_datetime(med_rev['YearMonth'])  # Corrected variable name
med_rev.set_index('YearMonth', inplace=True)

# Extract training and test data
train_data_med = med_rev.loc[:'2019-06-01', 'Revenue']
test_data_med = med_rev.loc['2019-07-01':'2019-12-01', 'Revenue']

model = ExponentialSmoothing(train_data_med, seasonal='add', seasonal_periods=9)  # 9 bagus
fit_model = model.fit()

# Make predictions on the test set
predictions = fit_model.forecast(len(test_data))

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(train_data_med.index, train_data_med, label='Training Data')
plt.plot(test_data_med.index, test_data_med, label='Test Data')
plt.plot(test_data_med.index, predictions, label='Predictions', color='red')
plt.title('Monthly Median Enterprise Sales Exponential Smoothing Forecasting with Seasonality')
plt.legend()
plt.show()

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100

print(f'Mean Absolute Percentage Error for median model (MAPE): {mape:.2f}%')


"""
#---------------- Forcasting using ETS (Date) -------------------------
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Load the dataset
e_c_s = pd.read_csv("e_commerce_sales_dataset.csv", encoding="ISO-8859-1")

# Convert 'Date' column to datetime and set it as index
e_c_s['Date'] = pd.to_datetime(e_c_s['Date'])
e_c_s.set_index('Date', inplace=True)

# Extract training and test data
train_data = e_c_s.loc[:'2019-06-30', 'Revenue']
test_data = e_c_s.loc['2019-07-01':'2019-12-30', 'Revenue']

# Fit the Exponential Smoothing model with seasonal_periods
# Adjust the seasonal_periods parameter based on the frequency of your seasonality
model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=3)  # Example parameters, adjust as needed
fit_model = model.fit()

# Make predictions on the test set
predictions = fit_model.forecast(len(test_data))

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data, label='Training Data')
plt.plot(test_data.index, test_data, label='Test Data')
plt.plot(test_data.index, predictions, label='Predictions', color='red')
plt.title('Exponential Smoothing Forecasting with Seasonality')
plt.legend()
plt.show()
"""


