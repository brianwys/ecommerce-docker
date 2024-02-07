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

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

df = pd.read_csv("department_store_dataset.csv", encoding="ISO-8859-1")


# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])
# Extract the month and year from the 'Date' column
df['YearMonth'] = df['Date'].dt.to_period("M")
# Drop the original 'Date' column
df = df.drop(columns=['Date'])

#df['Date'] = pd.to_datetime(df['Date'])
#df['quarter'] = df['Date'].dt.to_period("Q")


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

"""
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


"""