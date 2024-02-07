#------------- Forcasting using ETS E_S/enterprise sales (Month AVG) ----------
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