import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import os

# Specify the directory containing the CSV files
directory_path = r'C:\Users\MRaihan\Downloads\har_code\kpi_analysis\Data_Processing\department-based-on-month\med'

# Get a list of all files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

# Create an empty dictionary to store the results for each dataset
results_dict = {}
mape_dict = {}  # New dictionary to store MAPE values
all_plots = []  # List to store all plots

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
    plt.plot(train_data.index, train_data, label='Training Data', color='b')
    plt.plot(test_data.index, test_data, label='Test Data', color='b')
    plt.plot(test_data.index, final_predictions, label='Final Predictions', color='g')
    plt.title(f'Monthly Sales Exponential Smoothing Forecasting for {variable_name} with Seasonality\nMAPE: {best_mape:.2f}%')
    plt.legend()

    # Save the plot as an image file
    plot_filename = f'forecast_plot_{variable_name}.png'
    plot_filepath = os.path.join(directory_path, plot_filename)
    plt.savefig(plot_filepath)
    all_plots.append(plot_filepath)  # Add the filepath to the list
    plt.close()

# Print the results for each dataset
for key, result in results_dict.items():
    print(f'\nResults for {key}:')
    print(f'Best MAPE: {result["mape"]:.2f}%')
    print(f'Best Parameters: {result["best_params"]}')

# Print the MAPE values in the separate dictionary
print('\nMAPE Values:')
for key, mape_value in mape_dict.items():
    print(f'{key}: {mape_value:.2f}%')

# Print all plot filepaths
print('\nAll Plot Filepaths:')
for plot_filepath in all_plots:
    print(plot_filepath)


#a = results_dict["med_revenue_by_month_outside_salesdataset"]
#b = a["final_predictions"]

# Assuming b is a DataFrame
#b.to_csv('final_pred.csv', index=False)

