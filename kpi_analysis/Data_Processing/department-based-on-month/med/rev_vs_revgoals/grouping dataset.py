import pandas as pd

df = pd.read_csv("Dataset.csv", encoding="ISO-8859-1")

# Group by the 'department' column
grouped_df = df.groupby('Department')

# Create a dictionary to store department datasets
department_data = {}

# Iterate over groups and store them in the dictionary
for department_name, group in grouped_df:
    department_data[department_name] = group.copy()

# Accessing the dataset for the "Outside Sales" department
outside_sales = department_data["Outside Sales"]
outside_sales.drop(columns=["Region"], inplace=True)

# Group by the 'Department', 'Region', and 'YearMonth' columns
grouped_df = outside_sales.groupby(['Department', 'YearMonth'])

# Calculate the sum for the specified columns
columns_to_sum = ['Revenue', 'Revenue Goal', 'Margin', 'Margin Goal', 'Sales Quantity', 'Customers']
data_sum_per_region_date_outside_sales = grouped_df[columns_to_sum].sum().reset_index()
#data_sum_per_region_date_outside_sales.to_csv("data_sum_per_region_date_outside_sales.csv", index=False)
"""
columns_to_sum].median().reset_index()
rev_med = rev_med.rename(columns={'Revenue': 'Revenue_Median'})
rev_median = rev_med[['YearMonth', 'Revenue_Median']]

rev_goal_med = rev_med.rename(columns={'Revenue Goal': 'Revenue_Goal_Median'})
# Select specific columns from the DataFrames
rev_median = rev_med[[rev_med = grouped_df['YearMonth', 'Revenue_Median']]
rev_goal_median = rev_goal_med[['YearMonth', 'Revenue_Goal_Median']]
# Now, result_df contains the sum of specified columns grouped by 'Department', 'Region', and 'YearMonth'

data_sum_per_region_date['KPi Achieved'] = ['NO' if r < rg else 'YES' for r, rg in zip(data_sum_per_region_date["Revenue"], data_sum_per_region_date["Revenue Goal"])]
rev_goal_median.to_csv("rev_goal_median.csv",index=False)
rev_median.to_csv("rev_med.csv", index=False)
data_sum_per_region_date.to_csv('data_sum_per_region_date.csv', index=False)
"""
#----------------------------- modelling Before Covid --------------------------------------
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import os

# Specify the directory containing the CSV files
directory_path = r'C:\Users\LENOVO\Downloads\IBM\kpi_analysis\Data_Processing\department-based-on-month\med\rev_vs_revgoals'

# Get a list of all files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('data_sum_per_region_date_outside_sales.csv')]

# Create an empty dictionary to store the results for each dataset
results_dict = {}
mape_dict = {}  # New dictionary to store MAPE values

# Hyperparameter grid for grid search
param_grid = {
    'seasonal_periods': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],  # Adjust based on seasonality in your data
    'trend': ['add', 'mul', None],
    'seasonal': ['add', 'additive', None]
}

# Iterate through the CSV files
for csv_file in csv_files:
    # Construct the variable name based on the file name
    variable_name = os.path.splitext(csv_file)[0].lower().replace(' ', '_')
    
    # Read the CSV file
    dataset = pd.read_csv(os.path.join(directory_path, csv_file), encoding="ISO-8859-1")

    # Extract the 'Revenue' column for training and testing
    train_data = dataset['Revenue'].loc[:29]
    test_data = dataset['Revenue'].loc[30:35]
    #test_data = dataset['Revenue'].loc[0:35]

    # Hyperparameter tuning using grid search
    best_mape = float('inf')
    best_params = {}

    for seasonal_periods in param_grid['seasonal_periods']:
        for trend in param_grid['trend']:
            for seasonal in param_grid['seasonal']:
                model = ExponentialSmoothing(train_data, seasonal=seasonal, seasonal_periods=seasonal_periods, trend=trend)
                fit_model = model.fit()
                predictions = fit_model.forecast(len(test_data))
                mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100

                if mape < best_mape:
                    best_mape = mape
                    best_params = {'seasonal_periods': seasonal_periods, 'trend': trend, 'seasonal': seasonal}

    # Use the best parameters to train the final model
    final_model = ExponentialSmoothing(train_data, **best_params)
    final_fit_model = final_model.fit()

    # Make predictions on the test set
    final_predictions = final_fit_model.forecast(len(test_data))

    # Store the results in the dictionary for the current dataset
    results_dict[variable_name] = {
        'train_data': train_data,
        'test_data': test_data,
        'final_predictions': final_predictions,
        'mape': best_mape,
        'best_params': best_params
    }

    # Store the MAPE value in the separate dictionary
    mape_dict[variable_name] = best_mape

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Test Data')
    plt.plot(test_data.index, final_predictions, label='Final Predictions', color='red')
    plt.title(f'Monthly Sales Exponential Smoothing Forecasting for {variable_name} with Seasonality\nMAPE: {best_mape:.2f}%')
    plt.legend()
    plt.show()

# Print the results for each dataset
for key, result in results_dict.items():
    print(f'\nResults for {key}:')
    print(f'Best MAPE: {result["mape"]:.2f}%')
    print(f'Best Parameters: {result["best_params"]}')

# Print the MAPE values in the separate dictionary
print('\nMAPE Values:')
for key, mape_value in mape_dict.items():
    print(f'{key}: {mape_value:.2f}%')
    
#a = results_dict["med_revenue_by_month_outside_salesdataset"]
#b = a["final_predictions"]

# Assuming b is a DataFrame
#b.to_csv('final_pred.csv', index=False)

#-------------------------- After Covid ---------------------------------------
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import os

# Specify the directory containing the CSV files
directory_path = r'C:\Users\LENOVO\Downloads\IBM\kpi_analysis\Data_Processing\department-based-on-month\med\rev_vs_revgoals'

# Get a list of all files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('data_sum_per_region_date_outside_sales.csv')]

# Create an empty dictionary to store the results for each dataset
results_dict = {}
mape_dict = {}  # New dictionary to store MAPE values

# Hyperparameter grid for grid search
param_grid = {
    'seasonal_periods': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],  # Adjust based on seasonality in your data
    'trend': ['add', 'add', None],
    'seasonal': ['add', 'additive', None]
}

# Iterate through the CSV files
for csv_file in csv_files:
    # Construct the variable name based on the file name
    variable_name = os.path.splitext(csv_file)[0].lower().replace(' ', '_')
    
    # Read the CSV file
    dataset = pd.read_csv(os.path.join(directory_path, csv_file), encoding="ISO-8859-1")

    # Extract the 'Revenue' column for training and testing
    train_data = pd.concat([dataset['Revenue'].loc[:38], dataset['Revenue'].loc[41:65]])
    test_data = dataset['Revenue'].loc[66:]
    #test_data = dataset['Revenue'].loc[0:35]

    # Hyperparameter tuning using grid search
    best_mape = float('inf')
    best_params = {}

    for seasonal_periods in param_grid['seasonal_periods']:
        for trend in param_grid['trend']:
            for seasonal in param_grid['seasonal']:
                model = ExponentialSmoothing(train_data, seasonal=seasonal, seasonal_periods=seasonal_periods, trend=trend)
                fit_model = model.fit()
                predictions = fit_model.forecast(len(test_data))
                mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100

                if mape < best_mape:
                    best_mape = mape
                    best_params = {'seasonal_periods': seasonal_periods, 'trend': trend, 'seasonal': seasonal}

    # Use the best parameters to train the final model
    final_model = ExponentialSmoothing(train_data, **best_params)
    final_fit_model = final_model.fit()

    # Make predictions on the test set
    final_predictions = final_fit_model.forecast(len(test_data))

    # Store the results in the dictionary for the current dataset
    results_dict[variable_name] = {
        'train_data': train_data,
        'test_data': test_data,
        'final_predictions': final_predictions,
        'mape': best_mape,
        'best_params': best_params
    }

    # Store the MAPE value in the separate dictionary
    mape_dict[variable_name] = best_mape

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Test Data')
    plt.plot(test_data.index, final_predictions, label='Final Predictions', color='red')
    plt.title(f'Monthly Sales Exponential Smoothing Forecasting for {variable_name} with Seasonality\nMAPE: {best_mape:.2f}%')
    plt.legend()
    plt.show()

# Print the results for each dataset
for key, result in results_dict.items():
    print(f'\nResults for {key}:')
    print(f'Best MAPE: {result["mape"]:.2f}%')
    print(f'Best Parameters: {result["best_params"]}')

# Print the MAPE values in the separate dictionary
print('\nMAPE Values:')
for key, mape_value in mape_dict.items():
    print(f'{key}: {mape_value:.2f}%')
    
#a = results_dict["med_revenue_by_month_outside_salesdataset"]
#b = a["final_predictions"]

# Assuming b is a DataFrame
#b.to_csv('final_pred.csv', index=False)

#--------------------- SARIMA -------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

# Specify the directory containing the CSV files
directory_path = r'C:\Users\LENOVO\Downloads\IBM\kpi_analysis\Data_Processing\department-based-on-month\med\rev_vs_revgoals'

# Get a list of all files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('data_sum_per_region_date_outside_sales.csv')]

# Create an empty dictionary to store the results for each dataset
results_dict_sarima = {}
mape_dict_sarima = {}  # New dictionary to store MAPE values

# Define the hyperparameter grid for grid search
p_values = [1, 2, 3]  # AR order
d_values = [1]  # differencing order
q_values = [1, 2, 3]  # MA order
seasonal_p_values = [1]  # seasonal AR order
seasonal_d_values = [1]  # seasonal differencing order
seasonal_q_values = [1]  # seasonal MA order
seasonal_periods = [12]  # seasonality (assuming monthly data)

# Perform grid search
param_grid = product(p_values, d_values, q_values, seasonal_p_values, seasonal_d_values, seasonal_q_values, seasonal_periods)

# Iterate through the CSV files
for csv_file in csv_files:
    # Construct the variable name based on the file name
    variable_name = os.path.splitext(csv_file)[0].lower().replace(' ', '_')
    
    # Read the CSV file
    dataset = pd.read_csv(os.path.join(directory_path, csv_file), encoding="ISO-8859-1")

    # Extract the 'Revenue' column for training and testing
    train_data = dataset['Revenue'].loc[41:65]
    test_data = dataset['Revenue'].loc[66:]
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_data_normalized = scaler.fit_transform(train_data.values.reshape(-1, 1))


    # Initialize variables for best model
    best_mape = float('inf')
    best_params = None

    # Hyperparameter tuning using grid search
    for params in param_grid:
        order = params[:3]  # ARIMA order
        seasonal_order = params[3:]  # Seasonal order

        model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
        fit_model = model.fit(disp=False, method='newton')

        # Make predictions on the test set
        sarima_predictions = fit_model.get_forecast(steps=len(test_data))
        sarima_mean = sarima_predictions.predicted_mean
        mape_sarima = np.mean(np.abs((test_data - sarima_mean) / test_data)) * 100

        # Update best model if current MAPE is lower
        if mape_sarima < best_mape:
            best_mape = mape_sarima
            best_params = {'order': order, 'seasonal_order': seasonal_order}

    # Use the best parameters to train the final model
    final_model = SARIMAX(train_data, order=best_params['order'], seasonal_order=best_params['seasonal_order'])
    final_fit_model = final_model.fit(disp=False)

    # Make predictions on the test set
    final_predictions = final_fit_model.get_forecast(steps=len(test_data)).predicted_mean

    # Store the results in the dictionary for the current dataset
    results_dict_sarima[variable_name] = {
        'train_data': train_data,
        'test_data': test_data,
        'sarima_predictions': final_predictions,
        'mape_sarima': best_mape,
        'best_params': best_params
    }

    # Store the MAPE value in the separate dictionary
    mape_dict_sarima[variable_name] = best_mape

    # If the MAPE is below 10%, consider it a good model
    if best_mape < 10:
        # Plotting the results
        plt.figure(figsize=(12, 6))
        plt.plot(train_data.index, train_data, label='Training Data')
        plt.plot(test_data.index, test_data, label='Test Data')
        plt.plot(test_data.index, final_predictions, label='SARIMA Predictions', color='red')
        plt.title(f'Monthly Sales SARIMA Forecasting for {variable_name}\nMAPE: {best_mape:.2f}%')
        plt.legend()
        plt.show()

# Print the results for each dataset
for key, result in results_dict_sarima.items():
    print(f'\nResults for {key}:')
    print(f'SARIMA MAPE: {result["mape_sarima"]:.2f}%')
    print(f'SARIMA Best Parameters: {result["best_params"]}')

# Print the MAPE values in the separate dictionary
print('\nSARIMA MAPE Values:')
for key, mape_value in mape_dict_sarima.items():
    print(f'{key}: {mape_value:.2f}%')

#------------------------------------------- XGboost---------------------------
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load your dataset
xgb_data = pd.read_csv("data_sum_per_region_date_outside_sales.csv")

# Assuming the first four columns are features (X) and the last column is the label (y)
X = xgb_data  # Use iloc for integer-location based indexing
x.drop()
y = xgb_data.iloc[:, 3]  # Assuming the last column is the label

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an XGBoost classifier
model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(set(y)))

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
