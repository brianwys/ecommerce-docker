import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

df = pd.read_csv("department_store_dataset.csv", encoding="ISO-8859-1")


# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])
# Extract the month and year from the 'Date' column
df['YearMonth'] = df['Date'].dt.to_period("M")
# Drop the original 'Date' column
df = df.drop(columns=['Date'])

df['Date'] = pd.to_datetime(df['Date'])
df['quarter'] = df['Date'].dt.to_period("Q")


#df = df.drop(columns=['Date'])
#a = df["Department"].unique()
#b= len(a)
#c = df["Region"].unique()
#d = len(c)

jenis_barang = {'name': ['EletrÃ´nicos', 'VestuÃ¡rio', 'AcessÃ³rios', 'Casa', 'Brinquedo', 'Esportes', 'Papelaria']}
jenis_barang_baru = ['Jawa', 'Sumatra', 'Kalimantan', 'Papua', 'Sulawesi', 'NTB', 'Bali']
df['Region'] = [jenis_barang_baru[i % len(jenis_barang_baru)] for i in range(len(df))]

# Sample DataFrame
data = {'Name': ['LetÃ\xadcia Nascimento', 'Ana Sousa', 'Gustavo Martins', 'Beatriz Santos',
                 'Camila Lima', 'Thiago Barbosa', 'LetÃ\xadcia Ribeiro', 'Enzo Nascimento',
                 'Guilherme Santos', 'VitÃ³ria Ribeiro', 'Julia AraÃºjo', 'Lucas Rodrigues',
                 'Thiago Carvalho', 'Mateus Barbosa', 'Camila Carvalho', 'Diego Cardoso',
                 'Jorge Santos', 'Raphael Silva', 'Caroline Reis']}

# Department names
department_names = ['Sales Operations', 'Inside Sales', 'Outside Sales', 'Account Management',
                    'Business Development', 'Sales Enablement', 'Channel Sales', 'Key Accounts',
                    'Sales Analytics', 'Customer Success', 'Sales Training', 'Strategic Partnerships',
                    'Sales Support', 'Retail Sales', 'E-commerce Sales', 'Enterprise Sales',
                    'International Sales', 'Sales Administration', 'Sales Engineering']


df['Department'] = [department_names[i % len(department_names)] for i in range(len(df))]

df['KPi Achieved'] = ['NO' if r < rg else 'YES' for r, rg in zip(df["Revenue"], df["Revenue Goal"])]
df.to_csv("Dataset.csv", index=False)
print(df)


# Create variables for each department
for department in department_names:
    department_df = df[df.Department == department]
    #exec(f"{department.lower().replace(' ', '_').replace('-', '_')} = department_df")
    #department_df.to_csv(f"{department.lower().replace(' ', '_').replace('-', '_')}_dataset.csv", index=False)

#---- Loop for reading the dataset and calculating the mean and median from sales data for all departments-----

# Create an empty dictionary to store the datasets
datasets = {}
avg_revenue_by_month_dict = {}
med_revenue_by_month_dict = {}

# Iterate through the department names
for department in department_names:
    # Construct the variable name based on the department
    variable_name = department.lower().replace(' ', '_') + 'dataset'
    
    # Read the CSV file and store it in the dictionary
    datasets[variable_name] = pd.read_csv(f"{department.lower().replace(' ', '_')}_dataset.csv", encoding="ISO-8859-1")
    
    # Calculate the average revenue by month for the current department
    avg_revenue_by_month = datasets[variable_name].groupby('YearMonth')['Revenue'].mean().reset_index()
   
   # Store the result in the dictionary
    avg_revenue_by_month_dict[variable_name] = avg_revenue_by_month

    med_revenue_by_month = datasets[variable_name].groupby('YearMonth')['Revenue'].median().reset_index()

    # Store the result in the dictionary
    med_revenue_by_month_dict[variable_name] = med_revenue_by_month
    
    #avg_revenue_by_month.to_csv(f"avg_revenue_by_month_{variable_name}.csv", index=False)
    #med_revenue_by_month.to_csv(f"med_revenue_by_month_{variable_name}.csv", index=False)











#----------------------------- Save to CSV if needed --------------------------

#avg_revenue_by_month_i_s.to_csv("average_revenue_by_month_i_s.csv", index=False)
#avg_revenue_by_month_e_s.to_csv("average_revenue_by_month_e_s.csv", index=False)
#med_revenue_by_month_e_s.to_csv("median_revenue_by_month_e_s.csv", index=False)
#med_revenue_by_month_i_s.to_csv("median_revenue_by_month_i_s.csv", index=False)
"""

datasets = {}
avg_revenue_by_month_dict = {}
med_revenue_by_month_dict = {}
forecast_results_dict = {}

# Iterate through the department names
for department in department_names:
    # Construct the variable name based on the department
    variable_name = department.lower().replace(' ', '_') + 'dataset'
    
    # Read the CSV file and store it in the dictionary
    datasets[variable_name] = pd.read_csv(f"{department.lower().replace(' ', '_')}_dataset.csv", encoding="ISO-8859-1")
    
    # Calculate the average revenue by month for the current department
    avg_revenue_by_month = datasets[variable_name].groupby('YearMonth')['Revenue'].mean().reset_index()
    
    # Store the result in the dictionary
    avg_revenue_by_month_dict[variable_name] = avg_revenue_by_month
    
    # ------------ Forcasting using ETS for the current department ----------
    avg_rev = avg_revenue_by_month
    avg_rev['YearMonth'] = pd.to_datetime(avg_rev['YearMonth'])
    avg_rev.set_index('YearMonth', inplace=True)
    
    # Extract training and test data
    train_data = avg_rev.loc[:'2019-06-01', 'Revenue']
    test_data = avg_rev.loc['2019-07-01':'2019-12-01', 'Revenue']
    
    model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=5)  # 9 is good
    fit_model = model.fit()
    
    # Make predictions on the test set
    predictions = fit_model.forecast(len(test_data))
    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Test Data')
    plt.plot(test_data.index, predictions, label='Predictions', color='red')
    plt.title(f'Monthly Average {department} Sales Exponential Smoothing Forecasting with Seasonality')
    plt.legend()
    plt.show()
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
    print(f'Mean Absolute Percentage Error for {department} (MAPE): {mape:.2f}%')
    
    # Store the forecasting results in the dictionary
    forecast_results_dict[variable_name] = predictions


"""
"""

#----------- Model dataset dari tahun 2017 sampai dengan 2019 pertengahan ----------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot
from pmdarima import auto_arima

# Load the dataset
e_c_s = pd.read_csv("e_commerce_sales_dataset.csv", encoding="ISO-8859-1")

# Convert 'Date' column to datetime and set it as index
e_c_s['Date'] = pd.to_datetime(e_c_s['Date'])
e_c_s.set_index('Date', inplace=True)

# Extract training and test data
train_data = e_c_s.loc[:'2019-06-30', 'Revenue']  # Adjust the date based on your dataset
test_data = e_c_s.loc['2019-07-01':'2019-12-30', 'Revenue']  # Adjust the date based on your dataset

# Auto ARIMA
model = auto_arima(train_data, seasonal=False, stepwise=True, suppress_warnings=True)#disini bisa di taruh True atau False untuk menjadikan model Auto Arima atau Sarima

# Fit the model
model.fit(train_data)

# Forecast
forecast, conf_int = model.predict(n_periods=len(test_data), return_conf_int=True)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data, label='Training Data')
plt.plot(test_data.index, test_data, label='Actual Test Data', color='blue')
plt.plot(test_data.index, forecast, label='Forecast', color='red')
plt.fill_between(test_data.index, conf_int[:, 1], conf_int[:, 0], color='pink', alpha=0.3, label='Confidence Interval')
plt.title('Auto ARIMA Forecasting')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(test_data, forecast)
print(f'Mean Squared Error: {mse}')
"""

"""
# Split the data using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=10)
for train_index, test_index in tscv.split(train_data):
    train_data_split, test_data_split = train_data.iloc[train_index], train_data.iloc[test_index]
    

    # Fit ARIMA model on training data
    order = (10, 1, 0)  # Replace with the appropriate order based on ACF and PACF plots
    model = ARIMA(train_data_split, order=order)
    results = model.fit()

    # Forecast on test data
    forecast_steps = len(test_data_split)
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean

    # Evaluate the model using Mean Squared Error
    mse = mean_squared_error(test_data_split, forecast_mean)
    print(f'Mean Squared Error: {mse}')

    # Plot the original data and the forecast
    plt.plot(train_data_split, label='Training Data')
    plt.plot(test_data_split.index, forecast_mean, label='Forecast', color='red')
    plt.plot(test_data_split.index, test_data_split, label='Actual', color='green')
    plt.title("ARIMA Forecast with scikit-learn TimeSeriesSplit")
    plt.legend()
    plt.show()
"""
"""
e_c_s = pd.read_csv("e_commerce_sales_dataset.csv", encoding="ISO-8859-1")
train_data = e_c_s.loc[:910,:] 
#e_s_train = e_s.loc[:910,:] 
#i_s_train = i_s.loc[:910,:] 

test_data = e_c_s.loc[910:1094,:] 

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

e_c_s['Date'] = pd.to_datetime(e_c_s['Date'])
e_c_s.set_index('Date', inplace=True)

# Generate synthetic time series data
np.random.seed(42)
data = np.cumsum(np.random.normal(size=100))


# Split the data using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(data):
    train_data, test_data = data[train_index], data[test_index]

    # Fit ARIMA model on training data
    order = (1, 1, 1)  # Replace with the appropriate order based on ACF and PACF plots
    model = ARIMA(train_data, order=order)
    results = model.fit()

    # Forecast on test data
    forecast_steps = len(test_data)
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean

    # Evaluate the model using Mean Squared Error
    mse = mean_squared_error(test_data, forecast_mean)
    print(f'Mean Squared Error: {mse}')

    # Plot the original data and the forecast
    plt.plot(train_data, label='Training Data')
    plt.plot(np.arange(len(train_data), len(train_data) + forecast_steps), forecast_mean, label='Forecast', color='red')
    plt.plot(np.arange(len(train_data), len(train_data) + forecast_steps), test_data, label='Actual', color='green')
    plt.title("ARIMA Forecast with scikit-learn TimeSeriesSplit")
    plt.legend()
    plt.show()

"""

"""
#----------- E-Commerce Sales Dataset -------------#
e_c_s['Date'] = pd.to_datetime(e_c_s['Date'])
e_c_s.set_index('Date', inplace=True)
#----------- Enterprise Sales ---------------------#
e_s['Date'] = pd.to_datetime(e_s['Date'])
e_s.set_index('Date', inplace=True)
#----------- Inside Sales -------------------------#
i_s['Date'] = pd.to_datetime(i_s['Date'])
i_s.set_index('Date', inplace=True) 

# Plot the multivariate time series
plt.figure(figsize=(12, 6))
plt.plot(e_c_s.index, e_c_s['Revenue'], label='E-Commerece Sales')
plt.plot(e_s.index, e_s['Revenue'], label='Enterprise Sales')
plt.plot(i_s.index, i_s['Revenue'], label='Inside Sales')
plt.xlabel('Times')
plt.ylabel('Revenue')
plt.title('Multivariate Time Series')
plt.legend()
plt.show()





#revenue_goals = df["Revenue Goal"].mean()
#df['Below_Goals'] = ['Yes' if revenue < revenue_goals else 'No' for revenue in df["Revenue"]]
"""

"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ------ data 2017 to 2022-------#
df = pd.read_csv("Dataset.csv", encoding="ISO-8859-1")

# Extract features (X) and target variable (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Extract the first 3 columns as a DataFrame
a = df.iloc[:, 0:3]

# Apply LabelEncoder to the target variable 'y'
le = LabelEncoder()
y = le.fit_transform(y)

# Apply LabelEncoder to each column in DataFrame 'a'
a = a.apply(LabelEncoder().fit_transform)

#Replace the original columns in df with the transformed columns in 'a'
df.iloc[:, 0:3] = a

#------- data 2019 to 2022------#
new_ds = df.loc[13870:,:]#ini dari 2019 to 2022
# Extract features (X) and target variable (y)
X = new_ds.iloc[:, :-1].values
y = new_ds.iloc[:, -1].values

# Extract the first 3 columns as a DataFrame
a = new_ds.iloc[:, 0:3]

# Apply LabelEncoder to the target variable 'y'
le = LabelEncoder()
y = le.fit_transform(y)

# Apply LabelEncoder to each column in DataFrame 'a'
a = a.apply(LabelEncoder().fit_transform)

#Replace the original columns in df with the transformed columns in 'a'
new_ds.iloc[:, 0:3] = a

# ------ Decoder -------#

# Apply inverse_transform to each column in DataFrame 'a'
#b = a.apply(lambda col: le.inverse_transform(col))

# Print the decoded DataFrame
#print(b)

# Apply inverse_transform to the target variable 'y'
#y = le.inverse_transform(y)

# Apply inverse_transform to each column in DataFrame 'a'
#a = a.apply(le.inverse_transform().fit_transform)

# Replace the original columns in df with the decoded columns in 'a'
#df.iloc[:, 0:3] = a

"""


