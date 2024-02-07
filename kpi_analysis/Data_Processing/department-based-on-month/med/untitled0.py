"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import os

# Specify the directory containing the CSV files
directory_path = r'C:\Users\LENOVO\Downloads\IBM\kpi_analysis\Data_Processing\department-based-on-month\med'
# Get a list of all files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]
# Create an empty dictionary to store the datasets
datasets = {}
# Iterate through the CSV files
for csv_file in csv_files:
    # Construct the variable name based on the file name
    variable_name = os.path.splitext(csv_file)[0].lower().replace(' ', '_') + '_dataset'
    
    # Read the CSV file and store it in the dictionary
    datasets[variable_name] = pd.read_csv(os.path.join(directory_path, csv_file), encoding="ISO-8859-1")

train_data_med = datasets['med_revenue_by_month_account_managementdataset_dataset']['Revenue'].loc[:29]
test_data_med = datasets['med_revenue_by_month_account_managementdataset_dataset']['Revenue'].loc[30:35]
# Convert 'YearMonth' to datetime and set it as the index for the correct dataset

model = ExponentialSmoothing(train_data_med, seasonal='add', seasonal_periods=15)  # 15 bagus
fit_model = model.fit()

# Make predictions on the test set
predictions = fit_model.forecast(len(test_data_med))

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(train_data_med.index, train_data_med, label='Training Data')
plt.plot(test_data_med.index, test_data_med, label='Test Data')
plt.plot(test_data_med.index, predictions, label='Predictions', color='red')
plt.title('Monthly Median Enterprise Sales Exponential Smoothing Forecasting with Seasonality')
plt.legend()
plt.show()

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((test_data_med - predictions) / test_data_med)) * 100

print(f'Mean Absolute Percentage Error for median model (MAPE): {mape:.2f}%')

"""


import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import os

# Specify the directory containing the CSV files
directory_path = r'C:\Users\LENOVO\Downloads\IBM\kpi_analysis\Data_Processing\department-based-on-month\med'

# Get a list of all files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

# Create an empty dictionary to store the datasets
datasets = {}

# Iterate through the CSV files
for csv_file in csv_files:
    # Construct the variable name based on the file name
    variable_name = os.path.splitext(csv_file)[0].lower().replace(' ', '_') + '_dataset'
    
    # Read the CSV file and store it in the dictionary
    datasets[variable_name] = pd.read_csv(os.path.join(directory_path, csv_file), encoding="ISO-8859-1")

# Iterate through each dataset in the dictionary
for key, dataset in datasets.items():
    # Extract the 'Revenue' column for training and testing
    train_data = dataset['Revenue'].loc[:29]
    test_data = dataset['Revenue'].loc[30:35]

    # Apply Exponential Smoothing model
    model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=15)
    fit_model = model.fit()

    # Make predictions on the test set
    predictions = fit_model.forecast(len(test_data))

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Test Data')
    plt.plot(test_data.index, predictions, label='Predictions', color='red')
    plt.title(f'Monthly Sales Exponential Smoothing Forecasting for {key} with Seasonality')
    plt.legend()
    plt.show()

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
    print(f'Mean Absolute Percentage Error for {key} model (MAPE): {mape:.2f}%')
 
    
#---------------------------------------------
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import os

# Specify the directory containing the CSV files
directory_path = r'C:\Users\LENOVO\Downloads\IBM\kpi_analysis\Data_Processing\department-based-on-month\med'

# Get a list of all files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

# Create dictionaries to store predicted values, test values, and MAPE for each dataset
predicted_values = {}
test_values = {}
mape_values = {}

# Create a dictionary to store the seasonal_periods for each dataset
seasonal_periods_dict = {'key1': 10, 'key2': 15, 'key3': 8}  # Add more datasets as needed

# Iterate through the CSV files
for csv_file in csv_files:
    # Construct the variable name based on the file name
    variable_name = os.path.splitext(csv_file)[0].lower().replace(' ', '_') + '_dataset'
    
    # Read the CSV file and store it in the dictionary
    datasets[variable_name] = pd.read_csv(os.path.join(directory_path, csv_file), encoding="ISO-8859-1")

# Iterate through each dataset in the dictionary
for key, dataset in datasets.items():
    # Extract the 'Revenue' column for training and testing
    train_data = dataset['Revenue'].loc[:29]
    test_data = dataset['Revenue'].loc[30:35]

    # Get the seasonal_periods value for the current dataset
    seasonal_periods = seasonal_periods_dict.get(key, 10)  # Default to 10 if not specified

    # Apply Exponential Smoothing model
    model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=seasonal_periods)
    fit_model = model.fit()

    # Make predictions on the test set
    predictions = fit_model.forecast(len(test_data))

    # Record predicted values and test values
    predicted_values[key] = predictions
    test_values[key] = test_data

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
    mape_values[key] = mape

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Test Data')
    plt.plot(test_data.index, predictions, label='Predictions', color='red')
    plt.title(f'Monthly Sales Exponential Smoothing Forecasting for {key} with Seasonality')
    plt.legend()
    plt.show()

    # Print and store the MAPE value
    print(f'Mean Absolute Percentage Error for {key} model (MAPE): {mape:.2f}%')

# Display the dictionary of MAPE values
print("MAPE values for each dataset:")
print(mape_values)
